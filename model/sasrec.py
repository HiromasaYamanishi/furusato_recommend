import itertools
import operator
import pickle
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import NeighborSampler
from torch_scatter.scatter import scatter
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
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

class SequenceDataset(Dataset):
    def __init__(self, train_items, indices, length):
        self.data = train_items
        self.indices = indices
        self.length = length
        
    def __getitem__(self, index):
        #print('indices', self.indices)
        #print('index', index)
        xs = []
        for i in index:
            data_tmp = self.data[self.indices[i]]
            xs.append(data_tmp[-min(50, len(data_tmp)):])
        length = self.length[self.indices[index]]
        length[length>=50] = 50
        return xs, length
    
    def __len__(self):
        return len(self.data)
    
#テキスト特徴量を最初のノード特徴量に大きく考慮したGraphSAGE
class SASRec(torch.nn.Module):
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

        self.user_features = torch.from_numpy(
            np.load(
                f"/home/yamanishi/project/furusato_recommend/data/cb/customer_feature_pad{suffix}.npy",
                allow_pickle=True,
            )
        )
        self.item_features = torch.from_numpy(
            np.load(
                f"/home/yamanishi/project/furusato_recommend/data/cb/product_feature_pad{suffix}.npy",
                allow_pickle=True,
            )
        )
        self.user_word_embedding = (
            torch.from_numpy(
                np.load(
                    f"/home/yamanishi/project/furusato_recommend/data/text/user_text_emb{suffix}.npy"
                )
            )
            .float()
            .to(config["device"])
        )
        self.item_word_embedding = (
            torch.from_numpy(
                np.load(
                    f"/home/yamanishi/project/furusato_recommend/data/text/product_text_emb{suffix}.npy"
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
                f"/home/yamanishi/project/furusato_recommend/data/cb/product_sentence_emb{suffix}.npy"
            )
        ).float()
        self.user_numeric_features = torch.from_numpy(
            np.load(
                f"/home/yamanishi/project/furusato_recommend/data/cb/user_numeric_feature{suffix}.npy"
            )
        ).float()
        self.item_numeric_features = torch.from_numpy(
            np.load(
                f"/home/yamanishi/project/furusato_recommend/data/cb/product_numeric_feature{suffix}.npy"
            )
        ).float()
        self.user_numeric_linear = torch.nn.Linear(
            self.user_numeric_features.size(1), self.latent_dim
        )
        self.item_numeric_linear = torch.nn.Linear(
            self.item_numeric_features.size(1), self.latent_dim
        )
        self.w_linears = nn.ModuleList()
        self.w_linears.append(
            nn.Linear(self.latent_dim * 2, self.latent_dim * 1)
        )  # TODO: change
        for i in range(self.num_layers - 1):
            self.w_linears.append(nn.Linear(self.latent_dim * 2, self.latent_dim * 1))
        self.v_linears = nn.ModuleList(
            [
                nn.Linear(self.latent_dim * 1, self.latent_dim * 1)
                for _ in range(self.num_layers)
            ]
        )
        self.optim = optim.Adam(self.parameters(), lr=config["lr"])
        self.user_proj = torch.nn.Linear(
            int(self.latent_dim * 3.5) + 300, self.latent_dim
        )
        self.item_proj = torch.nn.Linear(
            int(self.latent_dim * 3.5) + 300 + 768, self.latent_dim
        )
        self.device = self.config["device"]
        self.test_item_emb = None
        with open(f"./data/text/product_name_count{suffix}.pkl", "rb") as f:
            self.item_name = pickle.load(f)
        with open(f"./data/text/product_main_comment_count{suffix}.pkl", "rb") as f:
            self.item_main_comment = pickle.load(f)
        #with open(f"./data/text/product_review{suffix}.pkl", "rb") as f:
        #    self.item_review = pickle.load(f)
        with open(
            f"./data/text/product_main_list_comment_count{suffix}.pkl", "rb"
        ) as f:
            self.item_main_list_comment = pickle.load(f)

        with open(f"./data/text/user_name_count{suffix}.pkl", "rb") as f:
            self.user_name = pickle.load(f)
        with open(f"./data/text/user_main_comment_count{suffix}.pkl", "rb") as f:
            self.user_main_comment = pickle.load(f)
        with open(f"./data/text/user_main_list_comment_count{suffix}.pkl", "rb") as f:
            self.user_main_list_comment = pickle.load(f)
        # self.item_name=torch.from_numpy(np.load(f'./data/text/product_name_count{suffix}.npy'))
        # self.item_main_comment=torch.from_numpy(np.load(f'./data/text/product_main_comment_count{suffix}.npy'))
        # self.item_main_list_comment=torch.from_numpy(np.load(f'./data/text/product_main_list_comment_count{suffix}.npy'))
        # self.user_name_count = torch.from_numpy(np.load(f'./data/text/user_name_count{suffix}.npy'))
        # self.user_main_comment=torch.from_numpy(np.load(f'./data/text/user_main_comment_count{suffix}.npy'))
        # self.user_main_list_comment=torch.from_numpy(np.load(f'./data/text/user_main_list_comment_count{suffix}.npy'))
        print("loaded text vecs")
        self.vocab_num = self.item_name.shape[1]
        self.word_emb_dim = self.latent_dim // 2
        self.word_embedding = torch.nn.Embedding(
            num_embeddings=self.item_name.shape[1], embedding_dim=self.word_emb_dim
        )
        with open(f'./data/train_items_sequence{suffix}.pkl', 'rb') as f:
            self.train_items = pickle.load(f)
            
        self.sequence_length = torch.load(f'./data/train_sequence_length{suffix}.pt')
        
        self.dropout = nn.Dropout(0.2)  
        self.attn_layers = torch.nn.ModuleList([torch.nn.MultiheadAttention(self.latent_dim, 8, batch_first=True) for _ in range(self.num_layers)])
        self.attn_norm_layers = torch.nn.ModuleList([torch.nn.LayerNorm(self.latent_dim) for _ in range(self.num_layers)])
        self.ffn_norm_layers = torch.nn.ModuleList([torch.nn.LayerNorm(self.latent_dim) for _ in range(self.num_layers)])
        self.ffn_layers = torch.nn.ModuleList([torch.nn.Linear(self.latent_dim, self.latent_dim) for _ in range(self.num_layers)])
        self.item_linears = torch.nn.ModuleList([torch.nn.Linear(self.latent_dim, self.latent_dim) for _ in range(self.num_layers-1)])
        self.item_last_proj = torch.nn.Linear(self.latent_dim, self.latent_dim*1)
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
        for i, w in enumerate(self.w_linears):
            if i == self.num_layers - 1:
                nn.init.xavier_uniform_(w.weight)
            else:
                nn.init.xavier_uniform_(w.weight, gain=gain)
            nn.init.zeros_(w.bias)

    def get_text_embedding(self, index, mode="user"):
        if mode == "user":
            name = self.user_name[index.tolist()].tocoo()
            main_comment = self.user_main_comment[index.tolist()].tocoo()
            main_comment_list = self.user_main_list_comment[index.tolist()].tocoo()
        elif mode == "item":
            name = self.item_name[index.tolist()].tocoo()
            main_comment = self.item_main_comment[index.tolist()].tocoo()
            main_comment_list = self.item_main_list_comment[index.tolist()].tocoo()
            #review = self.item_review[index.tolist()].tocoo()

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
        '''
        if mode=='item':
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
        '''
        #if mode=='user':
        if True:
            text_embedding = torch.cat([name_out, comment_out, comment_list_out], dim=1)
        #elif mode=='item':
        #    text_embedding = torch.cat([name_out, comment_out, comment_list_out, review_out], dim=1)
        # text_embedding = torch.cat([name_embedding, comment_embedding], dim=1)
        return text_embedding

    def get_initial_user_emb(self, user):
        # user: CPU
        numeric_embedding = self.user_numeric_linear(
            self.user_numeric_features[user].to(self.device)
        )
        id_embedding = self.user_id_embedding(user.to(self.device))
        text_embedding = self.get_text_embedding(user, mode="user")
        word_embedding = self.user_word_embedding[user].to(self.device)
        feature_embedding = torch.mean(
            self.user_feature_embedding(self.user_features[user].to(self.device)), dim=1
        )
        feature_embedding = torch.cat(
            [numeric_embedding, text_embedding, word_embedding, feature_embedding],
            dim=1,
        )
        feature_embedding = self.user_proj(feature_embedding)
        user_embedding = torch.cat([id_embedding, feature_embedding], dim=1)
        return feature_embedding  # TODO: change

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
        feature_embedding = torch.cat(
            [
                numeric_embedding,
                text_embedding,
                sentence_embedding,
                word_embedding,
                feature_embedding,
            ],
            dim=1,
        )
        feature_embedding = self.item_proj(feature_embedding)
        item_embedding = torch.cat([id_embedding, feature_embedding], dim=1)
        return feature_embedding  # TODO: change

    def get_initial_emb(self, index):
        user_index, item_index = (index < self.n_user).to(self.device), (
            index >= self.n_user
        ).to(self.device)
        user, item = index[user_index], index[item_index]
        emb = torch.zeros((len(index), self.latent_dim * 1)).to(
            self.device
        )  # TODO: change
        # print(user, item)
        emb[user_index] = self.get_initial_user_emb(user)
        emb[item_index] = self.get_initial_item_emb(item - self.n_user)
        return emb
    
    def oneblock(self, x, layer):
        T = x.shape[1]
        attn_mask = torch.triu(torch.full((T, T), float('-inf')), diagonal=1)
        init_x = x
        x = self.attn_norm_layers[layer](x)
        x = self.attn_layers[layer](x, x, x, attn_mask=attn_mask.to(self.device))[0]
        x = self.dropout(x)
        x = (init_x + x).relu()
        init_x = x
        x = self.ffn_norm_layers[layer](x)
        x = self.ffn_layers[layer](x)
        x = init_x + self.dropout(x)
        return x

    def forward_user(self, x, length):
        # neighbors: [batch(item), batch*neighbor_size(user), batch*neighbor_size^2(item)]
        #          or [batch(user), batch*neighbor_size(item), batch*neighbor_size^2(user)]
        # offsets: [batch, batch*neighbor_size, batch*neighbor_size*neighbor_size]
        T = x.shape[1]
        attn_mask = torch.triu(torch.full((T, T), float('-inf')), diagonal=1)
        for i in range(self.num_layers):
            x = self.oneblock(x, i)
        out = []
        for xx, l in zip(x, length.squeeze(0)):
            out.append(torch.mean(xx[:torch.tensor(l).to(self.device)], axis=0))
            #out.append(torch.cat([xx[torch.tensor([l-1]).to(self.device)].squeeze(0), torch.mean(xx[:torch.tensor(l).to(self.device)], axis=0)]))
        out = torch.stack(out)
        #print(out.shape)
        return out
    
    def forward_item(self, x):
        for i in range(self.num_layers-1):
            x = self.item_linears[i](x)
            x = x.relu()
            
        x = self.item_last_proj(x)
        return x

    def loss(
        self, user_emb: torch.Tensor, pos_emb: torch.Tensor, neg_emb: torch.Tensor
    ):
        pos_scores = torch.sum(torch.mul(user_emb, pos_emb), dim=1)
        neg_scores = torch.sum(torch.mul(user_emb, neg_emb), dim=1)
        all_param = 0
        for k,v in self.named_parameters():
            if 'emb' in k:
                all_param += all_param + v.norm(2)
        all_param = all_param / user_emb.size(0)
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        loss = loss + all_param * self.config["decay"]
        return loss

    def OneEpoch(self, user, pos, neg):
        #pos = pos + self.n_user
        #neg = neg + self.n_user
        # print(self.edge_index)
        # print(user, pos, neg)
        #print('user', user)
        dataset = SequenceDataset(self.train_items, user, self.sequence_length)
        sampler = torch.utils.data.sampler.BatchSampler(
            torch.utils.data.SequentialSampler(torch.arange(len(user))),
            batch_size=self.config['bpr_batch_size'],
            drop_last=False)
        dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler)

        total_batch = len(user) // self.config["bpr_batch_size"] + 1
        aver_loss = 0
        for i,(sequence_data, length) in tqdm(enumerate(dataloader)):
            item_embedding = self.get_initial_item_emb(torch.arange(self.m_item))
            sequence_item_embedding = []
            for s in sequence_data:
                sequence_item_embedding.append(item_embedding[s].squeeze(0))
            #print(sequence_item_embedding)
            sequence_item_embedding = pad_sequence(sequence_item_embedding, batch_first=True, padding_value=0)
            user_emb = self.forward_user(sequence_item_embedding, length)
            pos_tmp = pos[i*self.config["bpr_batch_size"]: min(len(pos), (i+1)*self.config["bpr_batch_size"])]
            neg_tmp = neg[i*self.config["bpr_batch_size"]: min(len(neg), (i+1)*self.config["bpr_batch_size"])]
            pos_emb = self.forward_item(item_embedding[pos_tmp])
            neg_emb = self.forward_item(item_embedding[neg_tmp])
            #print(user_emb.shape, pos_emb.shape, neg_emb.shape)
            #print(user_emb.shape, pos_emb.shape)
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
        item_initial_emb = self.get_initial_item_emb(torch.arange(self.m_item))
        dataset = SequenceDataset(self.train_items, users, self.sequence_length)
        sampler = torch.utils.data.sampler.BatchSampler(
            torch.utils.data.SequentialSampler(torch.arange(len(users))),
            batch_size=self.config['test_u_batch_size'],
            drop_last=False)
        dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler)
        for i,(sequence_data, length) in tqdm(enumerate(dataloader)):
            item_embedding = self.get_initial_item_emb(torch.arange(self.m_item))
            sequence_item_embedding = []
            for s in sequence_data:
                sequence_item_embedding.append(item_embedding[s].squeeze(0))
            #print(sequence_item_embedding)
            sequence_item_embedding = pad_sequence(sequence_item_embedding, batch_first=True, padding_value=0)

            if i==0:break
        user_emb = self.forward_user(sequence_item_embedding, length)
        item_emb = self.forward_item(item_initial_emb)
        #print(user_emb.shape, item_emb.shape)
        # print(user_x.shape, item_x.shape)
        print(user_emb.shape)
        rating = torch.matmul(user_emb, item_emb.T)
        return rating
    
    
