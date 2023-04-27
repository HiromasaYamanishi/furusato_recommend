import itertools
import operator
import pickle
from collections import deque

import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn.conv import SAGEConv
from torch_geometric.nn.conv.sage_conv2 import SAGEConv2
from torch_scatter.scatter import scatter
import torchtext
from tqdm import tqdm

import world
from neighbor_sampling import uniform_neighbors
from utils import minibatch


def compute_offsets(neighbor_lens):
    cumsum = list(itertools.accumulate(neighbor_lens, operator.add))
    return [0] + cumsum[:-1]


def get_indice_offset(l):
    indice, lens = [], []
    for _ in l:
        indice.extend(_)
        lens.append(len(_))
    return indice, compute_offsets(lens)

class BagOfWords(nn.Module):
    def __init__(self, field, hidden_dims):
        super().__init__()

        self.emb = nn.Embedding(
            len(field.vocab.itos), hidden_dims,
            padding_idx=field.vocab.stoi[field.pad_token])
        #nn.init.xavier_uniform_(self.emb.weight)

    def forward(self, x, length):
        return self.emb(x).sum(1) / length.unsqueeze(1).float()

#テキスト特徴量を最初のノード特徴量に大きく考慮したGraphSAGE
#テキスト特徴量を最初のノード特徴量に大きく考慮したGraphSAGE
class FastSAGE(torch.nn.Module):
    def __init__(self, config, dataset):
        super().__init__()
        suffix = config["suffix"]
        self.dataset = dataset
        self.n_user = self.dataset.n_user
        self.m_item = self.dataset.m_item
        trainUser, trainItem = self.dataset.trainUser, self.dataset.trainItem
        self.edge_index = torch.cat(
            [
                torch.stack(
                    [torch.tensor(trainUser), torch.tensor(trainItem) + self.n_user],
                    dim=0,
                ),
                torch.stack(
                    [torch.tensor(trainItem) + self.n_user, torch.tensor(trainUser)],
                    dim=0,
                ),
            ],
            dim=1,
        )
        self.config = config
        self.latent_dim = self.config["recdim"]
        self.num_layers = self.config["layer"]
        self.num_neighbors = self.config["num_neighbors"]
        self.dropout = nn.Dropout(0.2)
        self.user_oc = torch.zeros(self.n_user)
        self.item_oc = torch.zeros(self.m_item)
        for u, i in zip(trainUser, trainItem):
            self.user_oc[u]+=1
            self.item_oc[i]+=1

        self.user_features = torch.from_numpy(
            np.load(
                f"/home/yamanishi/project/furusato_recommend/data/cb/{suffix}/customer_feature_pad{suffix}.npy",
                allow_pickle=True,
            )
        )
        self.item_features = torch.from_numpy(
            np.load(
                f"/home/yamanishi/project/furusato_recommend/data/cb/{suffix}/product_feature_pad{suffix}.npy",
                allow_pickle=True,
            )
        )
        self.user_word_embedding = (
            torch.from_numpy(
                np.load(
                    f"/home/yamanishi/project/furusato_recommend/data/text/{suffix}/user_text_emb{suffix}.npy"
                )
            )
            .float()
            .to(config["device"])
        )
        self.item_word_embedding = (
            torch.from_numpy(
                np.load(
                    f"/home/yamanishi/project/furusato_recommend/data/text/{suffix}/product_text_emb{suffix}.npy"
                )
            )
            .float()
            .to(config["device"])
        )
        self.user_feature_num = self.user_features.shape[1]
        self.item_feature_num = self.item_features.shape[1]
        self.user_feature_indices, self.user_offsets = get_indice_offset(
            self.user_features
        )
        self.item_feature_indices, self.item_offsets = get_indice_offset(
            self.item_features
        )
        self.user_feature_num = max([max(ufe) for ufe in self.user_features]) + 1
        self.item_feature_num = max([max(ife) for ife in self.item_features]) + 1
        self.user_id_embedding = torch.nn.Embedding(
            num_embeddings=self.n_user, embedding_dim=self.latent_dim
        )
        self.item_id_embedding = torch.nn.Embedding(
            num_embeddings=self.m_item, embedding_dim=self.latent_dim
        )
        self.item_feature_embedding = torch.nn.Embedding(
            num_embeddings=self.item_feature_num, embedding_dim=self.latent_dim
        )
        self.user_feature_embedding = torch.nn.Embedding(
            num_embeddings=self.user_feature_num, embedding_dim=self.latent_dim
        )
        self.item_sentence_embedding = torch.from_numpy(
            np.load(
                f"/home/yamanishi/project/furusato_recommend/data/cb/{suffix}/product_sentence_emb{suffix}.npy"
            )
        ).float()
        self.user_numeric_features = torch.from_numpy(
            np.load(
                f"/home/yamanishi/project/furusato_recommend/data/cb/{suffix}/user_numeric_feature{suffix}.npy"
            )
        ).float()
        self.item_numeric_features = torch.from_numpy(
            np.load(
                f"/home/yamanishi/project/furusato_recommend/data/cb/{suffix}/product_numeric_feature{suffix}.npy"
            )
        ).float()
        print('loaded numeric feature')
        self.user_numeric_linear = torch.nn.Linear(
            self.user_numeric_features.size(1), self.latent_dim
        )
        self.item_numeric_linear = torch.nn.Linear(
            self.item_numeric_features.size(1), self.latent_dim
        )
        #self.w_linears = nn.ModuleList()
        #self.w_linears.append(
        #    nn.Linear(self.latent_dim * 2, self.latent_dim * 1)
        #)  # TODO: change
        #for i in range(self.num_layers - 1):
        #    self.w_linears.append(nn.Linear(self.latent_dim * 2, self.latent_dim * 1))
        #self.v_linears = nn.ModuleList(
        #    [
        #        nn.Linear(self.latent_dim * 1, self.latent_dim * 1)
        #        for _ in range(self.num_layers)
        #    ]
        #)
        self.optim = optim.Adam(self.parameters(), lr=config["lr"])
        proj_dim = {'n': self.latent_dim,
                    'c': self.latent_dim,
                    't': int(self.latent_dim*1.5),
                    'w': 300,
                    's': 768,
                    'r': int(self.latent_dim*0.5)}
        
        user_proj_dim = 0
        for f in self.config['user_feature']:
            user_proj_dim+=proj_dim[f]
        item_proj_dim = 0
        for f in self.config['item_feature']:
            item_proj_dim+=proj_dim[f]
            
        self.user_proj = torch.nn.Linear(
            user_proj_dim , self.latent_dim
        )
        self.item_proj = torch.nn.Linear(
            item_proj_dim, self.latent_dim
        )
        self.device = self.config["device"]
        self.test_item_emb = None
        with open(f"./data/text/{suffix}/product_name_count{suffix}.pkl", "rb") as f:
            self.item_name = pickle.load(f)
        with open(f"./data/text/{suffix}/product_main_comment_count{suffix}.pkl", "rb") as f:
            self.item_main_comment = pickle.load(f)
        #with open(f"./data/text/product_review{suffix}.pkl", "rb") as f:
        #    self.item_review = pickle.load(f)
        self.item_review = []
        with open(
            f"./data/text/{suffix}/product_main_list_comment_count{suffix}.pkl", "rb"
        ) as f:
            self.item_main_list_comment = pickle.load(f)

        with open(f"./data/text/{suffix}/user_name_count{suffix}.pkl", "rb") as f:
            self.user_name = pickle.load(f)
        with open(f"./data/text/{suffix}/user_main_comment_count{suffix}.pkl", "rb") as f:
            self.user_main_comment = pickle.load(f)
        with open(f"./data/text/{suffix}/user_main_list_comment_count{suffix}.pkl", "rb") as f:
            self.user_main_list_comment = pickle.load(f)
        print("loaded text vecs")
        self.vocab_num = self.item_name.shape[1]
        self.word_emb_dim = self.latent_dim // 2
        self.word_embedding = torch.nn.Embedding(
            num_embeddings=self.item_name.shape[1], embedding_dim=self.word_emb_dim
        )
        self.convs = torch.nn.ModuleList()
        for i in range(self.num_layers):
            self.convs.append(SAGEConv2(self.latent_dim, self.latent_dim))

        self.init_parameters()
        
    
    def init_parameters(self):
        gain = nn.init.calculate_gain("relu")
        gain = 0.1
        nn.init.xavier_uniform_(
            self.item_feature_embedding.weight,
        )
        nn.init.xavier_uniform_(
            self.user_feature_embedding.weight,
        )
        nn.init.xavier_uniform_(
            self.word_embedding.weight,
        )
        nn.init.normal_(
            self.user_id_embedding.weight,
        )
        nn.init.normal_(
            self.item_id_embedding.weight,
        )
        nn.init.xavier_uniform_(self.user_numeric_linear.weight)
        nn.init.constant_(self.user_numeric_linear.bias, 0)
        nn.init.xavier_uniform_(self.item_numeric_linear.weight)
        nn.init.constant_(self.item_numeric_linear.bias, 0)
        for i, conv in enumerate(self.convs):
            if i == self.num_layers - 1:
                nn.init.xavier_uniform_(conv.lin_l.weight)
            else:
                nn.init.xavier_uniform_(conv.lin_l.weight, gain=gain)
            nn.init.zeros_(conv.lin_l.bias)

    def get_text_embedding(self, index, mode="user"):
        if mode == "user":
            name = self.user_name[index.tolist()].tocoo()
            main_comment = self.user_main_comment[index.tolist()].tocoo()
            main_comment_list = self.user_main_list_comment[index.tolist()].tocoo()
        elif mode == "item":
            name = self.item_name[index.tolist()].tocoo()
            main_comment = self.item_main_comment[index.tolist()].tocoo()
            main_comment_list = self.item_main_list_comment[index.tolist()].tocoo()
        #FIXME:これは同じことを3回やっているので統一できないか
        #FIXME:列と行を取り出し, scatterし, text embeddingを取り出す関数に実装し直す
        name_source, name_target = name.col, name.row
        # print(name_source)
        name_source_word_embedding = self.word_embedding(
            torch.from_numpy(name_source).to(self.device)
        )
        name_out = torch.zeros((len(index), self.word_emb_dim)).to(self.device)
        # print(name_source_word_embedding, torch.from_numpy(name_target).to(self.device), name_out)
        name_out = scatter(
            name_source_word_embedding,
            torch.from_numpy(name_target).long().to(self.device),
            dim=0,
            out=name_out,
            reduce="mean",
        )

        comment_source, comment_target = main_comment.col, main_comment.row
        comment_source_word_embedding = self.word_embedding(
            torch.from_numpy(comment_source).to(self.device)
        )
        comment_out = torch.zeros((len(index), self.word_emb_dim)).to(self.device)
        comment_out = scatter(
            comment_source_word_embedding,
            torch.from_numpy(comment_target).long().to(self.device),
            dim=0,
            out=comment_out,
            reduce="mean",
        )

        comment_list_source, comment_list_target = (
            main_comment_list.col,
            main_comment_list.row,
        )
        comment_list_source_word_embedding = self.word_embedding(
            torch.from_numpy(comment_list_source).to(self.device)
        )
        comment_list_out = torch.zeros((len(index), self.word_emb_dim)).to(self.device)
        comment_list_out = scatter(
            comment_list_source_word_embedding,
            torch.from_numpy(comment_list_target).long().to(self.device),
            dim=0,
            out=comment_list_out,
            reduce="mean",
        )
        text_embedding = [name_out, comment_out, comment_list_out]
        
        if mode=='item' and 'r' in self.config['item_feature']:
            review = self.item_review[index.tolist()].tocoo()
            review_source, review_target = review.col, review.row
            # print(review_source)
            review_source_word_embedding = self.word_embedding(
                torch.from_numpy(review_source).to(self.device)
            )
            review_out = torch.zeros((len(index), self.word_emb_dim)).to(self.device)
            # print(review_source_word_embedding, torch.from_numpy(review_target).to(self.device), review_out)
            review_out = scatter(
                review_source_word_embedding,
                torch.from_numpy(review_target).long().to(self.device),
                dim=0,
                out=review_out,
                reduce="mean",
            )
            text_embedding.append(review_out)
        text_embedding = torch.cat(text_embedding, dim=1)
        return text_embedding

    def get_initial_user_emb(self, user):
        # user: CPU
        numeric_embedding = self.user_numeric_linear(
            self.user_numeric_features[user].to(self.device)
        )
        text_embedding = self.get_text_embedding(user, mode="user")
        word_embedding = self.user_word_embedding[user].to(self.device)
        feature_embedding = torch.mean(
            self.user_feature_embedding(self.user_features[user].to(self.device)), dim=1
        )
        
        user_embedding = []
        if 'n' in self.config['user_feature']:
            user_embedding.append(numeric_embedding)
            
        if 't' in self.config['user_feature']:
            user_embedding.append(text_embedding)
            
        if 'w' in self.config['user_feature']:
            user_embedding.append(word_embedding)
            
        if 'c' in self.config['user_feature']:
            user_embedding.append(feature_embedding)
                
        user_embedding = torch.cat(user_embedding, dim=1)
        
        user_embedding = self.user_proj(user_embedding)
        if self.config['cold_start']:
            cold_index = user<10000
            feature_embedding[cold_index] = 0
            
        return user_embedding  # TODO: change

    def get_initial_item_emb(self, item):
        # item: CPU
        numeric_embedding = self.item_numeric_linear(
            self.item_numeric_features[item].to(self.device)
        )
        id_embedding = self.item_id_embedding(item.to(self.device))
        text_embedding = self.get_text_embedding(item, mode="item")
        sentence_embedding = self.item_sentence_embedding[item].to(self.device)
        word_embedding = self.item_word_embedding[item].to(self.device)
        feature_embedding = torch.mean(
            self.item_feature_embedding(self.item_features[item].to(self.device)), dim=1
        )
        
        item_embedding = []
        if 'n' in self.config['item_feature']:
            item_embedding.append(numeric_embedding)
            
        if 't' in self.config['item_feature']:
            item_embedding.append(text_embedding)
            
        if 'w' in self.config['item_feature']:
            item_embedding.append(word_embedding)
            
        if 'c' in self.config['item_feature']:
            item_embedding.append(feature_embedding)
            
        if 's' in self.config['item_feature']:
            item_embedding.append(sentence_embedding)
                
        item_embedding = torch.cat(item_embedding, dim=1)
        
        item_embedding = self.item_proj(item_embedding)
        return item_embedding # TODO: change

    def get_initial_emb(self, index):
        user_index, item_index = (index < self.n_user), (index >= self.n_user)
        user, item = index[user_index], index[item_index]
        emb = torch.zeros((len(index), self.latent_dim * 1)).to(
            self.device
        )  # TODO: change
        # print(user, item)
        emb[user_index] = self.get_initial_user_emb(user)
        emb[item_index] = self.get_initial_item_emb(item - self.n_user)
        return emb

    
    def propagate(self, x, edge_index, layer):
        x = self.convs[layer](x, edge_index.to(self.device))
        return x
    
    def forward(self, x, adjs):
        # neighbors: [batch(item), batch*neighbor_size(user), batch*neighbor_size^2(item)]
        #          or [batch(user), batch*neighbor_size(item), batch*neighbor_size^2(user)]
        # offsets: [batch, batch*neighbor_size, batch*neighbor_size*neighbor_size]
        for i, (edge_index, _, size) in enumerate(adjs):
            x = self.dropout(x)
            x = self.propagate(x, edge_index, i)
            x = x[: size[1]]
            if i != self.num_layers - 1:
                x = x.relu()
        return x

    def loss(
        self, user_emb: torch.Tensor, pos_emb: torch.Tensor, neg_emb: torch.Tensor
    ):
        pos_scores = torch.sum(torch.mul(user_emb, pos_emb), dim=1)
        neg_scores = torch.sum(torch.mul(user_emb, neg_emb), dim=1)
        all_param = 0
        for p in self.parameters():
            all_param += all_param + p.norm(2)
        all_param = all_param / user_emb.size(0)
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        loss = loss + all_param * self.config["decay"]
        return loss

    def OneEpoch(self, user, pos, neg):
        pos = pos + self.n_user
        neg = neg + self.n_user
        # print(self.edge_index)
        # print(user, pos, neg)
        user_loader = NeighborSampler(
            self.edge_index,
            node_idx=user,
            sizes=[self.num_neighbors for _ in range(self.num_layers)],
            batch_size=self.config["bpr_batch_size"],
            shuffle=False,
            num_workers=4,
        )
        pos_loader = NeighborSampler(
            self.edge_index,
            node_idx=pos,
            sizes=[self.num_neighbors for _ in range(self.num_layers)],
            batch_size=self.config["bpr_batch_size"],
            shuffle=False,
            num_workers=4,
        )
        neg_loader = NeighborSampler(
            self.edge_index,
            node_idx=neg,
            sizes=[self.num_neighbors, self.num_neighbors],
            batch_size=self.config["bpr_batch_size"],
            shuffle=False,
            num_workers=4,
        )
        total_batch = len(user) // self.config["bpr_batch_size"] + 1
        aver_loss = 0
        for (
            (user_batch_size, user_id, user_adjs),
            (pos_batch_size, pos_id, pos_adjs),
            (neg_batch_size, neg_id, neg_adjs),
        ) in tqdm(zip(user_loader, pos_loader, neg_loader)):
            assert user_batch_size == pos_batch_size == neg_batch_size
            # user_x, pos_x, neg_x = self.get_initial_train_emb(user_id, pos_id, neg_id)

            user_x = self.get_initial_emb(user_id)
            pos_x, neg_x = self.get_initial_emb(pos_id), self.get_initial_emb(neg_id)

            user_adjs = [user_adj.to(self.device) for user_adj in user_adjs]
            pos_adjs = [pos_adj.to(self.device) for pos_adj in pos_adjs]
            neg_adjs = [neg_adj.to(self.device) for neg_adj in neg_adjs]
            user_emb = self.forward(user_x, user_adjs)
            pos_emb = self.forward(pos_x, pos_adjs)
            neg_emb = self.forward(neg_x, neg_adjs)
            self.optim.zero_grad()
            loss = self.loss(user_emb, pos_emb, neg_emb)
            aver_loss += loss.detach().cpu()
            loss.backward()
            # loss.backward(retain_graph=True)
            del loss
            self.optim.step()
        aver_loss /= total_batch
        return aver_loss

    @torch.no_grad()
    def getUsersRating(self, users):
        user_initial_emb = self.get_initial_user_emb(torch.arange(self.n_user))
        item_initial_emb = self.get_initial_item_emb(torch.arange(self.m_item))

        user_x, item_x = user_initial_emb, item_initial_emb
        x = torch.cat([user_x, item_x], dim=0)
        for i in range(self.num_layers):
            x = self.propagate(x, self.edge_index, i)
            if i != self.num_layers - 1:
                x = x.relu()
        user_x, item_x = x[:self.n_user], x[self.n_user:]
        rating = torch.matmul(user_x[users], item_x.T)
        return rating
