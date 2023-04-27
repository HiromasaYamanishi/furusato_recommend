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

#テキスト特徴量を最初のノード特徴量に大きく考慮したGraphSAGE
class ASAGE(torch.nn.Module):
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
        
        self.user_attribute = torch.load(f'./data/attribute/user_attribute{suffix}.pt',)
        self.user_attribute_edge_index = torch.cat(
            [
                torch.stack(
                    [self.user_attribute[0], self.user_attribute[1] + self.n_user],
                    dim=0,
                ),
                torch.stack(
                    [self.user_attribute[1] + self.n_user, self.user_attribute[0]],
                    dim=0,
                ),
            ],
            dim=1,
        )
        self.item_attribute = torch.load(f'./data/attribute/product_attribute{suffix}.pt')
        self.item_attribute_edge_index = torch.cat(
            [
                torch.stack(
                    [self.item_attribute[0], self.item_attribute[1] + self.m_item],
                    dim=0,
                ),
                torch.stack(
                    [self.item_attribute[1] + self.m_item, self.item_attribute[0]],
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
        self.user_attr_embedding = torch.nn.Embedding(num_embeddings=self.user_attribute[1].max()+1, embedding_dim=self.latent_dim)
        self.item_attr_embedding = torch.nn.Embedding(num_embeddings=self.item_attribute[1].max()+1, embedding_dim=self.latent_dim)
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
        self.ssl_temp = 0.2
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
            self.item_attr_embedding.weight,
        )
        nn.init.xavier_uniform_(
            self.user_attr_embedding.weight,
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
    
    def get_initial_attr_emb(self, index, mode):
        if mode=='user':
            user_index, attr_index = (index < self.n_user).to(self.device), (
                index >= self.n_user
            ).to(self.device)
            user, attr = index[user_index], index[attr_index]
            emb = torch.zeros((len(index), self.latent_dim * 1)).to(
                self.device
            )  # TODO: change
            # print(user, item)
            emb[user_index] = self.get_initial_user_emb(user)
            emb[attr_index] = self.user_attr_embedding((attr-self.n_user).to(self.device))
        
        elif mode=='item':
            item_index, attr_index = (index < self.m_item).to(self.device), (
                index >= self.m_item
            ).to(self.device)
            item, attr = index[item_index], index[attr_index]
            emb = torch.zeros((len(index), self.latent_dim * 1)).to(
                self.device
            )  # TODO: change
            # print(item, item)
            emb[item_index] = self.get_initial_item_emb(item)
            emb[attr_index] = self.item_attr_embedding((attr-self.m_item).to(self.device))
            
        return emb
        
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

    def propagate(self, x, edge_index, layer, size=None):
        source_x = x[edge_index[0]]
        source_x = self.dropout(source_x)
        if size is not None:
            target_x = x[: size[1]]
        else:
            target_x = x
        out = torch.zeros((target_x.size(0), source_x.size(1))).to(self.device)
        aggr = scatter(source_x, edge_index[1], dim=0, out=out, reduce="mean")
        x = self.w_linears[layer](torch.cat([target_x, aggr], dim=1))
        if layer != self.num_layers - 1:
            x = x.relu()
        return x
        
    def forward(self, x, adjs):
        # neighbors: [batch(item), batch*neighbor_size(user), batch*neighbor_size^2(item)]
        #          or [batch(user), batch*neighbor_size(item), batch*neighbor_size^2(user)]
        # offsets: [batch, batch*neighbor_size, batch*neighbor_size*neighbor_size]
        x_init = x[: adjs[-1][-1][1]]
        for i, (edge_index, _, size) in enumerate(adjs):
            x = self.propagate(x, edge_index, i, size)
        return x
    

    def loss(
        self, user_emb: torch.Tensor, pos_emb: torch.Tensor, neg_emb: torch.Tensor,
            user_attr_emb, pos_attr_emb, neg_attr_emb
    ):
        pos_scores = torch.sum(torch.mul(user_emb, pos_emb), dim=1)
        neg_scores = torch.sum(torch.mul(user_emb, neg_emb), dim=1)
        pos_attr_scores = torch.sum(torch.mul(user_attr_emb, pos_attr_emb), dim=1)
        neg_attr_scores = torch.sum(torch.mul(user_attr_emb, neg_attr_emb), dim=1)
        all_param = 0
        for k,v in self.named_parameters():
            if 'attr_emb' in k:continue
            all_param += all_param + v.norm(2)
        all_param = all_param / user_emb.size(0)
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        attr_loss = torch.mean(torch.nn.functional.softplus(neg_attr_scores - pos_attr_scores))
        
        '''
        user_emb = F.normalize(user_emb, dim=1)
        user_attr_emb = F.normalize(user_attr_emb, dim=1)
        pos_emb = F.normalize(pos_emb, dim=1)
        pos_attr_emb = F.normalize(pos_attr_emb, dim=1)

        pos_ratings_user = torch.sum(torch.mul(user_emb, user_attr_emb), dim=1)
        pos_ratings_item = torch.sum(torch.mul(pos_emb, pos_attr_emb), dim=1)
        tot_ratings_user = torch.matmul(user_emb, 
                                        torch.transpose(user_attr_emb, 0, 1))        # [batch_size, num_users]
        tot_ratings_item = torch.matmul(pos_emb, 
                                        torch.transpose(pos_attr_emb, 0, 1))        # [batch_size, num_items]

        ssl_logits_user = tot_ratings_user - pos_ratings_user[:, None]                  # [batch_size, num_users]
        ssl_logits_item = tot_ratings_item - pos_ratings_item[:, None]                  # [batch_size, num_users]
        clogits_user = torch.logsumexp(ssl_logits_user / self.ssl_temp, dim=1)
        clogits_item = torch.logsumexp(ssl_logits_item / self.ssl_temp, dim=1)
        infonce_loss = torch.mean(clogits_user + clogits_item)
        '''
        #print(loss, attr_loss, infonce_loss)
        loss = loss + 1e-1* attr_loss+ all_param * self.config["decay"]
        return loss

    def OneEpoch(self, user, pos, neg):
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
            node_idx=pos+self.n_user,
            sizes=[self.num_neighbors for _ in range(self.num_layers)],
            batch_size=self.config["bpr_batch_size"],
            shuffle=False,
            num_workers=4,
        )
        neg_loader = NeighborSampler(
            self.edge_index,
            node_idx=neg+self.n_user,
            sizes=[self.num_neighbors, self.num_neighbors],
            batch_size=self.config["bpr_batch_size"],
            shuffle=False,
            num_workers=4,
        )
        user_attr_loader = NeighborSampler(
            self.user_attribute_edge_index,
            node_idx=user,
            sizes=[self.num_neighbors for _ in range(self.num_layers)],
            batch_size=self.config["bpr_batch_size"],
            shuffle=False,
            num_workers=4,
        )
        pos_attr_loader = NeighborSampler(
            self.item_attribute_edge_index,
            node_idx=pos,
            sizes=[self.num_neighbors for _ in range(self.num_layers)],
            batch_size=self.config["bpr_batch_size"],
            shuffle=False,
            num_workers=4,
        )
        neg_attr_loader = NeighborSampler(
            self.item_attribute_edge_index,
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
            (user_batch_size, user_attr_id, user_attr_adjs),
            (pos_batch_size, pos_attr_id, pos_attr_adjs),
            (neg_batch_size, neg_attr_id, neg_attr_adjs),
        ) in tqdm(zip(user_loader, pos_loader, neg_loader, user_attr_loader, pos_attr_loader, neg_attr_loader)):
            assert user_batch_size == pos_batch_size == neg_batch_size
            # user_x, pos_x, neg_x = self.get_initial_train_emb(user_id, pos_id, neg_id)

            user_x = self.get_initial_emb(user_id)
            pos_x, neg_x = self.get_initial_emb(pos_id), self.get_initial_emb(neg_id)
            
            user_attr_x = self.get_initial_attr_emb(user_attr_id, mode='user')
            pos_attr_x, neg_attr_x = self.get_initial_attr_emb(pos_attr_id, mode='item'), self.get_initial_attr_emb(neg_attr_id, mode='item')
            
            user_adjs = [user_adj.to(self.device) for user_adj in user_adjs]
            pos_adjs = [pos_adj.to(self.device) for pos_adj in pos_adjs]
            neg_adjs = [neg_adj.to(self.device) for neg_adj in neg_adjs]
            user_attr_adjs = [user_attr_adj.to(self.device) for user_attr_adj in user_attr_adjs]
            pos_attr_adjs = [pos_attr_adj.to(self.device) for pos_attr_adj in pos_attr_adjs]
            neg_attr_adjs = [neg_attr_adj.to(self.device) for neg_attr_adj in neg_attr_adjs]
            user_emb = self.forward(user_x, user_adjs)
            pos_emb = self.forward(pos_x, pos_adjs)
            neg_emb = self.forward(neg_x, neg_adjs)
            user_attr_emb = self.forward(user_attr_x, user_attr_adjs)
            pos_attr_emb = self.forward(pos_attr_x, pos_attr_adjs)
            neg_attr_emb = self.forward(neg_attr_x, neg_attr_adjs)
            self.optim.zero_grad()
            loss = self.loss(user_emb, pos_emb, neg_emb, user_attr_emb, pos_attr_emb, neg_attr_emb)
            aver_loss += loss.detach().cpu()
            loss.backward()
            # loss.backward(retain_graph=True)
            del loss
            self.optim.step()
        aver_loss /= total_batch
        return aver_loss

    @torch.no_grad()
    def getUsersRating(self, users):
        trainItem = torch.tensor(self.dataset.trainItem).to(self.device)
        trainUser = torch.tensor(self.dataset.trainUser).to(self.device)
        user_initial_emb = self.get_initial_user_emb(torch.arange(self.n_user))
        item_initial_emb = self.get_initial_item_emb(torch.arange(self.m_item))
        user_attr_emb = self.user_attr_embedding.weight
        item_attr_emb = self.item_attr_embedding.weight
        user_x, item_x = user_initial_emb, item_initial_emb
        x = torch.cat([user_x, item_x])
        for i in range(self.num_layers):
            x = self.propagate(x, self.edge_index.to(self.device), i)
        user_x_out, item_x_out = x[:self.n_user], x[self.n_user:]
        
        x_user_attr = torch.cat([user_x, user_attr_emb])
        for i in range(self.num_layers):
            x_user_attr = self.propagate(x_user_attr, self.user_attribute_edge_index.to(self.device), i)
        user_x_attr_out, _ = x_user_attr[:self.n_user], x_user_attr[self.n_user:]
        
        x_item_attr = torch.cat([item_x, item_attr_emb])
        for i in range(self.num_layers):
            x_item_attr = self.propagate(x_item_attr, self.item_attribute_edge_index.to(self.device), i)
        item_x_attr_out, _ = x_item_attr[:self.m_item], x_user_attr[self.m_item:]
        # print(user_x.shape, item_x.shape)
        rating = torch.matmul(user_x_out[users], item_x_out.T) + 0.1*torch.matmul(user_x_attr_out[users], item_x_attr_out.T)
        return rating
