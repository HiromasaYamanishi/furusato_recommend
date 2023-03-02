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
        
def compute_offsets(neighbor_lens):
    cumsum = list(itertools.accumulate(neighbor_lens, operator.add))
    return [0] + cumsum[:-1]

def get_indice_offset(l):
    indice, lens = [], []
    for _ in l:
        indice.extend(_)
        lens.append(len(_))
    return indice, compute_offsets(lens)
        
class GraphSAGE(torch.nn.Module):
    def __init__(self, config, dataset):
        super().__init__()
        self.dataset = dataset
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
        
    def get_initial_hiddens(self, neighbors, mode):
        initial_hiddens = []
        user_initial_embedding = self.user_proj(F.embedding_bag(torch.tensor(self.user_feature_indices).to(self.device),
                                                 self.user_feature_embeddings.weight,
                                                 offsets=torch.tensor(self.user_offsets).to(self.device),
                                                 mode='mean'))
        item_initial_embedding = self.item_proj(F.embedding_bag(torch.tensor(self.item_feature_indices).to(self.device),
                                                 self.item_feature_embeddings.weight,
                                                 offsets=torch.tensor(self.item_offsets).to(self.device),
                                                 mode='mean'))
        offset=0 if mode=='user' else 1
        for i in range(len(neighbors)):
            if offset%2:
                initial_hiddens.append(item_initial_embedding[neighbors[i]])
            else:
                initial_hiddens.append(user_initial_embedding[neighbors[i]])
            offset+=1
        return initial_hiddens

    def forward(self, neighbors, offsets, mode):
        hidden = self.get_initial_hiddens(neighbors, mode=mode)
        for layer in range(self.num_layers):
            w_linear = self.w_linears[layer]
            next_hidden = []
            depth = self.num_layers - layer
            for k in range(depth):
                current_embeds = self.dropout(hidden[k])
                neighbor_embeds = self.dropout(hidden[k+1])
                mean_neighbors = F.embedding_bag(
                    torch.arange(neighbor_embeds.shape[0]).to(self.device),
                    neighbor_embeds,
                    offsets = offsets[k+1].to(self.device),
                    mode='mean'
                )
                h = torch.cat([current_embeds, mean_neighbors], dim=1)
                if layer == self.num_layers - 1:
                    h = w_linear(h)
                else:
                    h = F.relu(w_linear(h))
                next_hidden.append(h)
            hidden = next_hidden
        return hidden[0]
            
    def loss(self, user_emb: torch.Tensor, pos_emb: torch.Tensor, neg_emb:torch.Tensor):
        pos_scores = torch.sum(torch.mul(user_emb, pos_emb), dim=1)
        neg_scores = torch.sum(torch.mul(user_emb, neg_emb), dim=1)
        all_param = 0
        for p in self.parameters():
            all_param+=all_param+p.norm(2).pow(2)
        loss = torch.mean(torch.nn.functional.softplus(neg_scores-pos_scores))
        loss = loss+all_param*self.config['decay']
        return loss
        
    def stageOne(self, user: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor):
        user_neighbors, user_offsets= self.sampling_neighbors(user, mode='user')
        pos_neighbors, pos_offsets = self.sampling_neighbors(pos, mode='item')
        neg_neighbors, neg_offsets = self.sampling_neighbors(neg, mode='item')
        user_emb = self.forward(user_neighbors, user_offsets, mode='user')
        pos_emb = self.forward(pos_neighbors, pos_offsets, mode='item')
        neg_emb = self.forward(neg_neighbors, neg_offsets, mode='item')
        self.optim.zero_grad()
        loss = self.loss(user_emb, pos_emb, neg_emb)
        loss.backward()
        self.optim.step()
        return loss
    
    @torch.no_grad()
    def getUsersRating(self, users):
        items = torch.arange(self.dataset.m_item)
        user_emb_all, item_emb_all = [], []
        user_neighbors, user_offsets= self.sampling_neighbors(users, mode='user')
        user_emb = self.forward(user_neighbors, user_offsets, mode='user')
        #    user_emb_all.append(user_emb)
        #user_emb = torch.cat(user_emb_all)
        for item in tqdm(minibatch(items)):
            item_neighbors, item_offsets= self.sampling_neighbors(item, mode='item')
            item_emb = self.forward(item_neighbors, item_offsets, mode='item')
            item_emb_all.append(item_emb)
        item_emb = torch.cat(item_emb_all)      
        rating = torch.matmul(user_emb, item_emb.T)
        return rating 
            
    
    def sampling_neighbors(self, nodes, mode):
        neighbors, offsets = [], []
        neighbors.append(torch.tensor(nodes))
        offsets.append(torch.arange(len(nodes)))
        for i in range(self.num_layers):
            neighbor, offset = uniform_neighbors(nodes, self.dataset, self.num_neighbors, mode)
            neighbors.append(torch.tensor(neighbor))
            offsets.append(torch.tensor(offset))
            nodes = neighbor
            if mode=='user':
                mode='item'
            else:mode='user'
        return neighbors, offsets
        
        
    
    
        
        
        
        
        
        
    
        
        
        
        
        