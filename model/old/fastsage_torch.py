import torch
import torch.nn as nn
import torch.nn.functional as F
from neighbor_sampling import uniform_neighbors
import itertools
import numpy as np
import operator
import torch.optim as optim
from utils import minibatch
from tqdm import tqdm
from torch_scatter.scatter import scatter
from collections import deque
from torch_geometric.loader import NeighborSampler
        
def compute_offsets(neighbor_lens):
    cumsum = list(itertools.accumulate(neighbor_lens, operator.add))
    return [0] + cumsum[:-1]

def get_indice_offset(l):
    indice, lens = [], []
    for _ in l:
        indice.extend(_)
        lens.append(len(_))
    return indice, compute_offsets(lens)
        
class FastSAGETorch(torch.nn.Module):
    def __init__(self, config, dataset):
        super().__init__()
        self.dataset = dataset
        self.n_user = self.dataset.n_user
        self.m_item = self.dataset.m_item
        trainUser, trainItem = self.dataset.trainUser, self.dataset.trainItem
        self.edge_index = torch.cat([torch.stack([torch.tensor(trainUser), torch.tensor(trainItem)+self.n_user], dim=0), torch.stack([torch.tensor(trainItem)+self.n_user, torch.tensor(trainUser)], dim=0)], dim=1)
        self.config = config
        self.latent_dim = self.config['recdim']
        self.num_layers = self.config['layer']
        self.num_neighbors = self.config['num_neighbors']
        self.dropout = nn.Dropout(0.2)
        
        self.user_features = np.load('/home/yamanishi/project/furusato_recommend/data/cb/customer_features.npy', allow_pickle=True)
        self.item_features = np.load('/home/yamanishi/project/furusato_recommend/data/cb/product_features.npy', allow_pickle=True)
        self.user_feature_indices, self.user_offsets = get_indice_offset(self.user_features)
        self.item_feature_indices, self.item_offsets = get_indice_offset(self.item_features)
        #self.user_feature_num = max([max(ufe) for ufe in self.user_features])
        #self.item_feature_num = max([max(ife) for ife in self.item_features])
        self.user_feature_num = 3207
        self.item_feature_num = 2094
        #print(self.user_feature_num)
        #print(self.item_feature_num)
        self.item_feature_embeddings = torch.nn.Embedding(num_embeddings=self.item_feature_num, embedding_dim=self.latent_dim)
        self.user_feature_embeddings = torch.nn.Embedding(num_embeddings=self.user_feature_num, embedding_dim=self.latent_dim)
        self.w_linears = nn.ModuleList([nn.Linear(self.latent_dim*2, self.latent_dim) for _ in range(self.num_layers)])
        self.init_parameters()
        self.optim = optim.Adam(self.parameters(), lr=config['lr'])
        self.user_proj = torch.nn.Linear(self.latent_dim, self.latent_dim)
        self.item_proj = torch.nn.Linear(self.latent_dim, self.latent_dim)
        self.device = self.config['device']
        self.test_item_emb = None
        
        
    def init_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.item_feature_embeddings.weight)
        nn.init.xavier_uniform_(self.user_feature_embeddings.weight)
        for i, w in enumerate(self.w_linears):
            if i == self.num_layers - 1:
                nn.init.xavier_uniform_(w.weight)
            else:
                nn.init.xavier_uniform_(w.weight, gain=gain)
            nn.init.zeros_(w.bias)
            
    def get_initial_embedding(self):
        user_initial_embedding = self.user_proj(F.embedding_bag(torch.tensor(self.user_feature_indices).to(self.device),
                                                 self.user_feature_embeddings.weight,
                                                 offsets=torch.tensor(self.user_offsets).to(self.device),
                                                 mode='mean'))
        item_initial_embedding = self.item_proj(F.embedding_bag(torch.tensor(self.item_feature_indices).to(self.device),
                                                 self.item_feature_embeddings.weight,
                                                 offsets=torch.tensor(self.item_offsets).to(self.device),
                                                 mode='mean'))
        
        return user_initial_embedding, item_initial_embedding
        
    
    
    def forward(self, x, adjs):
        #neighbors: [batch(item), batch*neighbor_size(user), batch*neighbor_size^2(item)]
        #          or [batch(user), batch*neighbor_size(item), batch*neighbor_size^2(user)]
        #offsets: [batch, batch*neighbor_size, batch*neighbor_size*neighbor_size]
        for i, (edge_index, _, size) in enumerate(adjs):
            source_x = x[edge_index[0]]
            target_x = x[:size[1]]
            out = torch.zeros((size[1], self.latent_dim)).to(self.device)
            aggr = scatter(source_x, edge_index[1], dim=0, out=out)
            x = self.w_linears[i](torch.cat([target_x, aggr], dim=1))
            if i!=self.num_layers-1:
                x = x.relu()
        return x

            
    def loss(self, user_emb: torch.Tensor, pos_emb: torch.Tensor, neg_emb:torch.Tensor):
        pos_scores = torch.sum(torch.mul(user_emb, pos_emb), dim=1)
        neg_scores = torch.sum(torch.mul(user_emb, neg_emb), dim=1)
        all_param = 0
        for p in self.parameters():
            all_param+=all_param+p.norm(2).pow(2)
        loss = torch.mean(torch.nn.functional.softplus(neg_scores-pos_scores))
        loss = loss+all_param*self.config['decay']
        return loss
    
    def OneEpoch(self, user, pos, neg):
        pos = pos + self.n_user
        neg = neg + self.n_user
        user_loader = NeighborSampler(self.edge_index, node_idx=user, sizes=[self.num_neighbors for _ in range(self.num_layers)],
                                      batch_size=self.config['bpr_batch_size'], shuffle=False, num_workers=16)
        pos_loader = NeighborSampler(self.edge_index, node_idx=pos, sizes=[self.num_neighbors for _ in range(self.num_layers)],
                                batch_size=self.config['bpr_batch_size'], shuffle=False, num_workers=16)
        neg_loader = NeighborSampler(self.edge_index, node_idx=neg, sizes=[self.num_neighbors, self.num_neighbors],
                                batch_size=self.config['bpr_batch_size'], shuffle=False, num_workers=16)
        total_batch = len(user)//self.config['bpr_batch_size'] + 1
        aver_loss = 0
        user_init_x, item_init_x = self.get_initial_embedding()
        x = torch.cat([user_init_x, item_init_x], dim=0)
        for (user_batch_size, user_id, user_adjs), (pos_batch_size, pos_id, pos_adjs), (neg_batch_size, neg_id, neg_adjs) in tqdm(zip(user_loader, pos_loader, neg_loader)):
            assert user_batch_size == pos_batch_size == neg_batch_size
            user_adjs = [user_adj.to(self.device) for user_adj in user_adjs]
            pos_adjs = [pos_adj.to(self.device) for pos_adj in pos_adjs]
            neg_adjs = [neg_adj.to(self.device) for neg_adj in neg_adjs]
            user_emb = self.forward(x[user_id], user_adjs)
            pos_emb = self.forward(x[pos_id], pos_adjs)
            neg_emb = self.forward(x[neg_id],neg_adjs)
            self.optim.zero_grad()
            loss = self.loss(user_emb, pos_emb, neg_emb)
            aver_loss+=loss
            loss.backward(retain_graph=True)
            self.optim.step()
        aver_loss/=total_batch
        return aver_loss
    
    
        
    @torch.no_grad()
    def getUsersRating(self, users):
        if self.config['inference'] == 'all':
            trainItem = torch.tensor(self.dataset.trainItem).to(self.device)
            trainUser =  torch.tensor(self.dataset.trainUser).to(self.device)
            user_intitial_emb, item_initial_emb = self.get_initial_embedding()
            user_x, item_x = user_intitial_emb, item_initial_emb
            for i in range(self.num_layers):
                user_out, item_out = torch.zeros_like(user_x), torch.zeros_like(item_x)
                user_x_all, item_x_all = user_x[trainUser], item_x[trainItem]
                user_x_agg = scatter(item_x_all, trainUser, out=user_out, dim=0)
                item_x_agg = scatter(user_x_all, trainItem, out=item_out, dim=0)
                user_x = self.w_linears[i](torch.cat([user_x, user_x_agg], dim=1))
                item_x = self.w_linears[i](torch.cat([item_x, item_x_agg], dim=1))
                if i!=self.num_layers-1:
                    item_x = item_x.relu()
                    user_x = user_x.relu()

            rating = torch.matmul(user_x[users], item_x.T)
            return rating 
        
        elif self.config['inference']=='sample':
            all_item = torch.arange(self.n_user, self.n_user+self.m_item)
            user_initial_emb, item_initial_emb = self.get_initial_embedding()
            x = torch.cat([user_initial_emb, item_initial_emb], dim=0)
            item_emb_all = []
            if self.test_item_emb==None:
                item_sampler = NeighborSampler(self.edge_index, node_idx=all_item, sizes=[self.num_neighbors for _ in range(self.num_layers)],
                            batch_size=self.config['test_u_batch_size'], shuffle=False, num_workers=16)
                for (item_batch_size, item_id, item_adjs) in item_sampler:
                    adj = [adj.to(self.device) for adj in item_adjs]
                    item_emb = self.forward(x[item_id], adj)
                    item_emb_all.append(item_emb)
                self.test_item_emb = torch.cat(item_emb_all, dim=0)
            
            user_sampler = NeighborSampler(self.edge_index, node_idx=all_item, sizes=[self.num_neighbors for _ in range(self.num_layers)],
                            batch_size=self.config['test_u_batch_size'], shuffle=False, num_workers=16)
            
            for (user_batch_size, user_id, user_adjs) in user_sampler:
                adj = [adj.to(self.device) for adj in user_adjs]
                user_emb = self.forward(x[user_id], adj)
                
            rating = torch.matmul(user_emb, self.test_item_emb.T)
            return rating
            
    
        
        
    
    
        
        
        
        
        
        
    
        
        
        
        
        