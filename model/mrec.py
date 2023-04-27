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
import time

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

class FactorizationMachine(torch.nn.Module):

    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix

#テキスト特徴量を最初のノード特徴量に大きく考慮したGraphSAGE
class MRec(torch.nn.Module):
    def __init__(self, config, dataset):
        super().__init__()
        suffix = config["suffix"]
        self.dataset = dataset
        self.n_user = self.dataset.n_user
        self.m_item = self.dataset.m_item
        trainUser, trainItem = self.dataset.trainUser, self.dataset.trainItem
        # Construct edge index
        # item index in this setting item index + user_num
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
        
        # count How many times user/item interacted
        self.user_oc = torch.zeros(self.n_user)
        self.item_oc = torch.zeros(self.m_item)
        for u, i in zip(trainUser, trainItem):
            self.user_oc[u]+=1
            self.item_oc[i]+=1
        init_time = time.time()
        if 'c' in self.config['user_feature']:
            self.user_features = torch.from_numpy(
                np.load(
                    f"/home/yamanishi/project/furusato_recommend/data/cb/{suffix}/customer_feature_pad{suffix}.npy",
                    allow_pickle=True,
                )
            )
        if 'c' in self.config['item_feature']:
            self.item_features = torch.from_numpy(
                np.load(
                    f"/home/yamanishi/project/furusato_recommend/data/cb/{suffix}/product_feature_pad{suffix}.npy",
                    allow_pickle=True,
                )
            )
        print('loaded category feature')
        if 'w' in self.config['user_feature']:
            self.user_word_embedding = (
                torch.from_numpy(
                    np.load(
                        f"/home/yamanishi/project/furusato_recommend/data/text/{suffix}/user_text_emb{suffix}.npy"
                    )
                )
                .float()
            )
        if 'w' in self.config['item_feature']:
            self.item_word_embedding = (
                torch.from_numpy(
                    np.load(
                        f"/home/yamanishi/project/furusato_recommend/data/text/{suffix}/product_text_emb{suffix}.npy"
                    )
                )
                .float()
            )
        print('loaded word embedding')
        print(time.time()-init_time)
        self.user_feature_num = 4798#max([max(ufe) for ufe in self.user_features]) + 1
        self.item_feature_num = 2762#max([max(ife) for ife in self.item_features]) + 1
        self.item_feature_embedding = torch.nn.Embedding(
            num_embeddings=self.item_feature_num, embedding_dim=self.latent_dim
        )
        self.user_feature_embedding = torch.nn.Embedding(
            num_embeddings=self.user_feature_num, embedding_dim=self.latent_dim
        )
        print('load category embedding')
        print(time.time()-init_time)

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
        print('loaded numeric', time.time()-init_time)
        if 'n' in self.config['user_feature']:
            self.user_numeric_linear = torch.nn.Linear(
                self.user_numeric_features.shape[1], self.latent_dim
            )
        if 'n' in self.config['item_feature']:
            self.item_numeric_linear = torch.nn.Linear(
                self.item_numeric_features.shape[1], self.latent_dim
            )
            
        if 'b' in self.config['user_feature']:
            self.user_bert_feature = torch.load(f'/home/yamanishi/project/furusato_recommend/data/text/{suffix}/customer_deberta_feature{suffix}.pt')
        
        if 'b' in self.config['item_feature']:
            self.item_bert_feature = torch.load(f'/home/yamanishi/project/furusato_recommend/data/text/{suffix}/product_deberta_feature{suffix}.pt')
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
        proj_dim = {'n': self.latent_dim,
                    'c': self.latent_dim,
                    't': int(self.latent_dim*1.5),
                    'w': 300,
                    's': 768,
                    'r': int(self.latent_dim*0.5),
                    'b': 768}
        
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
        self.user_mlp = torch.nn.ModuleList([torch.nn.Linear(user_proj_dim, self.latent_dim), 
                                             torch.nn.GELU(), 
                                             torch.nn.Linear(self.latent_dim, self.latent_dim)])
        
        self.item_mlp = torch.nn.ModuleList([torch.nn.Linear(item_proj_dim, self.latent_dim),
                                            torch.nn.GELU(),
                                            torch.nn.Linear(self.latent_dim, self.latent_dim)])
        self.device = self.config["device"]
        self.test_item_emb = None
        with open(f"./data/text/{suffix}/product_name_tfidf{suffix}.pkl", "rb") as f:
            self.item_name = pickle.load(f)
        with open(f"./data/text/{suffix}/product_main_comment_tfidf{suffix}.pkl", "rb") as f:
            self.item_main_comment = pickle.load(f)
        with open(
            f"./data/text/{suffix}/product_main_list_comment_tfidf{suffix}.pkl", "rb"
        ) as f:
            self.item_main_list_comment = pickle.load(f)
        if 'r' in self.config['item_feature']:
            with open(f"./data/text/{suffix}/product_review{suffix}.pkl", "rb") as f:
                self.item_review = pickle.load(f)
        if 's' in self.config['user_feature']:
            self.user_sentence_embedding=torch.load(f'/home/yamanishi/project/furusato_recommend/data/text/{suffix}/customer_sentence_feature{suffix}.pt')  
        if 's' in self.config['item_feature']:
            self.item_sentence_embedding= torch.from_numpy(np.load(f"/home/yamanishi/project/furusato_recommend/data/cb/{suffix}/product_sentence_emb{suffix}.npy"))

        with open(f"./data/text/{suffix}/user_name_count{suffix}.pkl", "rb") as f:
            self.user_name = pickle.load(f)
        with open(f"./data/text/{suffix}/user_main_comment_count{suffix}.pkl", "rb") as f:
            self.user_main_comment = pickle.load(f)
        with open(f"./data/text/{suffix}/user_main_list_comment_count{suffix}.pkl", "rb") as f:
            self.user_main_list_comment = pickle.load(f)
        
        # self.item_name=torch.from_numpy(np.load(f'./data/text/product_name_count{suffix}.npy'))
        # self.item_main_comment=torch.from_numpy(np.load(f'./data/text/product_main_comment_count{suffix}.npy'))
        # self.item_main_list_comment=torch.from_numpy(np.load(f'./data/text/product_main_list_comment_count{suffix}.npy'))
        # self.user_name_count = torch.from_numpy(np.load(f'./data/text/user_name_count{suffix}.npy'))
        # self.user_main_comment=torch.from_numpy(np.load(f'./data/text/user_main_comment_count{suffix}.npy'))
        # self.user_main_list_comment=torch.from_numpy(np.load(f'./data/text/user_main_list_comment_count{suffix}.npy'))
        print("loaded text vecs", time.time()-init_time)
        self.vocab_num = self.item_name.shape[1]
        self.word_emb_dim = self.latent_dim // 2
        self.word_embedding = torch.nn.Embedding(
            num_embeddings=self.item_name.shape[1], embedding_dim=self.word_emb_dim
        )
        

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
            
    def get_text_embedding_from_coo(self, word_coo, index):
        word_source, word_target = word_coo.col, word_coo.row
        source_word_embedding = self.word_embedding(
            torch.from_numpy(word_source).to(self.device)
        )
        
        word_out = torch.zeros((len(index), self.word_emb_dim)).to(self.device)
        word_out = scatter(
            source_word_embedding,
            torch.from_numpy(word_target).long().to(self.device),
            dim=0,
            out=word_out,
            reduce='mean'
        )
        return word_out
        
        

    def get_text_embedding(self, index, mode="user"):
        if mode == "user":
            name = self.user_name[index.tolist()].tocoo()
            main_comment = self.user_main_comment[index.tolist()].tocoo()
            main_comment_list = self.user_main_list_comment[index.tolist()].tocoo()
        elif mode == "item":
            name = self.item_name[index.tolist()].tocoo()
            main_comment = self.item_main_comment[index.tolist()].tocoo()
            main_comment_list = self.item_main_list_comment[index.tolist()].tocoo()

        name_embedding = self.get_text_embedding_from_coo(name, index)
        comment_embedding = self.get_text_embedding_from_coo(main_comment, index)
        list_comment_embedding = self.get_text_embedding_from_coo(main_comment_list, index)
        text_embedding = [name_embedding, comment_embedding, list_comment_embedding]
        
        if mode=='item' and 'r' in self.config['item_feature']:
            review = self.item_review[index.tolist()].tocoo()
            review_embedding = self.get_text_embedding_from_coo(review, index)
            text_embedding.append(review_embedding)
        text_embedding = torch.cat(text_embedding, dim=1)
        return text_embedding

    def get_initial_user_emb(self, user):
        # user: CPU
        
        user_embedding = []
        if 'n' in self.config['user_feature']:
            numeric_embedding = self.user_numeric_linear(
                self.user_numeric_features[user].to(self.device)
            )
            user_embedding.append(numeric_embedding)
            
        if 't' in self.config['user_feature']:
            text_embedding = self.get_text_embedding(user, mode="user")
            user_embedding.append(text_embedding)
            
        if 'w' in self.config['user_feature']:
            word_embedding = self.user_word_embedding[user].to(self.device)
            user_embedding.append(word_embedding)
            
        if 'c' in self.config['user_feature']:
            feature_embedding = self.user_feature_embedding(self.user_features[user].to(self.device))
            feature_mean_embedding = torch.mean(feature_embedding, dim=1)
            if self.config['factorization']:
                factorized_feature_embedding = self.factorization_machine(feature_embedding)
                feature_mean_embedding = torch.cat([feature_mean_embedding, factorized_feature_embedding], dim=1)

            user_embedding.append(feature_mean_embedding)
            
        if 'b' in self.config['user_feature']:
            bert_feature = self.user_bert_feature.to(self.device)[user]
            user_embedding.append(bert_feature)
            
        if 's' in self.config['user_feature']:
            sentence_embedding = self.user_sentence_embedding.to(self.device)[user]
            user_embedding.append(sentence_embedding)
            
                
        user_embedding = torch.cat(user_embedding, dim=1)
        
        for layer in self.user_mlp:
            user_embedding = layer(user_embedding)
            
        #user_embedding = self.user_proj(user_embedding)
        if self.config['cold_start']:
            cold_index = user<10000
            user_embedding[cold_index] = 0
            
        return user_embedding  # TODO: change

    def get_initial_item_emb(self, item):
        # item: CPU
        
        item_embedding = []
        if 'n' in self.config['item_feature']:
            numeric_embedding = self.item_numeric_linear(
                self.item_numeric_features[item].to(self.device)
            )
            item_embedding.append(numeric_embedding)
            
        if 't' in self.config['item_feature']:
            text_embedding = self.get_text_embedding(item, mode="item")
            item_embedding.append(text_embedding)
            
        if 'w' in self.config['item_feature']:
            word_embedding = self.item_word_embedding[item].to(self.device)
            item_embedding.append(word_embedding)
            
        if 'c' in self.config['item_feature']:
            feature_embedding = self.item_feature_embedding(self.item_features[item].to(self.device))
            feature_mean_embedding = torch.mean(feature_embedding, dim=1)
            if self.config['factorization']:
                factorized_feature_embedding = self.factorization_machine(feature_embedding)
                feature_mean_embedding = torch.cat([feature_mean_embedding, factorized_feature_embedding], dim=1)

            item_embedding.append(feature_mean_embedding)
            
        if 's' in self.config['item_feature']:
            sentence_embedding = self.item_sentence_embedding[item].to(self.device)
            item_embedding.append(sentence_embedding)
            
        if 'b' in self.config['item_feature']:
            bert_feature = self.user_bert_feature.to(self.device)[item]
            item_embedding.append(bert_feature)
                
        item_embedding = torch.cat(item_embedding, dim=1)
        
        for layer in self.item_mlp:
            item_embedding = layer(item_embedding)
        #item_embedding = self.item_proj(item_embedding)
        return item_embedding

    def get_initial_emb(self, index):
        user_index, item_index = (index < self.n_user), (index >= self.n_user)
        user, item = index[user_index], index[item_index]
        emb = torch.zeros((len(index), self.latent_dim * 1)).to(
            self.device
        ) 
        # print(user, item)
        emb[user_index] = self.get_initial_user_emb(user)
        emb[item_index] = self.get_initial_item_emb(item - self.n_user)
        return emb

    def forward(self, x, adjs):
        # neighbors: [batch(item), batch*neighbor_size(user), batch*neighbor_size^2(item)]
        #          or [batch(user), batch*neighbor_size(item), batch*neighbor_size^2(user)]
        # offsets: [batch, batch*neighbor_size, batch*neighbor_size*neighbor_size]
        for i, (edge_index, _, size) in enumerate(adjs):
            source_x = x[edge_index[0]]
            source_x = self.dropout(source_x)
            target_x = x[: size[1]]
            out = torch.zeros((size[1], source_x.size(1))).to(self.device)
            aggr = scatter(source_x, edge_index[1], dim=0, out=out, reduce="mean")
            x = self.w_linears[i](torch.cat([target_x, aggr], dim=1))
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
        #edge_indexにおいてitemのノードは+num_userされている
        #そのため, pos, negはそれぞれ+n_userする
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
        # Inference Code
        trainItem = torch.tensor(self.dataset.trainItem).to(self.device)
        trainUser = torch.tensor(self.dataset.trainUser).to(self.device)
        user_initial_emb = self.get_initial_user_emb(torch.arange(self.n_user))
        item_initial_emb = self.get_initial_item_emb(torch.arange(self.m_item))

        user_x, item_x = user_initial_emb, item_initial_emb
        
        for i in range(self.num_layers):
            user_aggr, item_aggr = torch.zeros_like(user_x), torch.zeros_like(item_x)
            #Use minibatch based edge iteration to prevent OOM
            for trainUsertmp, trainItemtmp in minibatch(trainUser, trainItem, batch_size=10000):
                user_x_tmp, item_x_tmp = user_x[trainUsertmp], item_x[trainItemtmp]
                user_div = self.user_oc[trainUsertmp.cpu()]
                item_div = self.item_oc[trainItemtmp.cpu()]
                item_x_tmp = item_x_tmp/user_div.unsqueeze(1).to(self.device)
                user_x_tmp = user_x_tmp/item_div.unsqueeze(1).to(self.device)
                user_out = torch.zeros_like(user_x)
                item_out = torch.zeros_like(item_x)
                user_out = scatter(
                    item_x_tmp, trainUsertmp, out=user_out, dim=0, reduce="sum"
                )
                item_out = scatter(
                    user_x_tmp, trainItemtmp, out=item_out, dim=0, reduce="sum"
                )
                user_aggr = user_aggr + user_out
                item_aggr = item_aggr + item_out
            user_x = self.w_linears[i](torch.cat([user_x, user_aggr], dim=1))
            item_x = self.w_linears[i](torch.cat([item_x, item_aggr], dim=1))
            if i != self.num_layers - 1:
                item_x = item_x.relu()
                user_x = user_x.relu()
        # print(user_x.shape, item_x.shape)
        rating = torch.matmul(user_x[users], item_x.T)
        return rating
