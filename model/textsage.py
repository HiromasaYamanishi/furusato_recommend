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
import world
import pickle
        
def compute_offsets(neighbor_lens):
    cumsum = list(itertools.accumulate(neighbor_lens, operator.add))
    return [0] + cumsum[:-1]

def get_indice_offset(l):
    indice, lens = [], []
    for _ in l:
        indice.extend(_)
        lens.append(len(_))
    return indice, compute_offsets(lens)
        
class TextSAGE(torch.nn.Module):
    def __init__(self, config, dataset):
        super().__init__()
        suffix = config['suffix']
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
        
        self.user_features = torch.from_numpy(np.load(f'/home/yamanishi/project/furusato_recommend/data/cb/customer_feature_pad{suffix}.npy', allow_pickle=True))
        self.item_features = torch.from_numpy(np.load(f'/home/yamanishi/project/furusato_recommend/data/cb/product_feature_pad{suffix}.npy', allow_pickle=True))
        self.user_word_embedding = torch.from_numpy(np.load(f'/home/yamanishi/project/furusato_recommend/data/text/user_text_emb{suffix}.npy')).float().to(config['device'])
        self.item_word_embedding  = torch.from_numpy(np.load(f'/home/yamanishi/project/furusato_recommend/data/text/product_text_emb{suffix}.npy')).float().to(config['device'])
        self.user_feature_num = self.user_features.shape[1]
        self.item_feature_num = self.item_features.shape[1]
        self.user_feature_indices, self.user_offsets = get_indice_offset(self.user_features)
        self.item_feature_indices, self.item_offsets = get_indice_offset(self.item_features)
        self.user_feature_num = max([max(ufe) for ufe in self.user_features])+1
        self.item_feature_num = max([max(ife) for ife in self.item_features])+1
        self.user_id_embeddings = torch.nn.Embedding(num_embeddings=self.n_user, embedding_dim=self.latent_dim)
        self.item_id_embeddings = torch.nn.Embedding(num_embeddings=self.m_item, embedding_dim=self.latent_dim)
        self.item_feature_embeddings = torch.nn.Embedding(num_embeddings=self.item_feature_num, embedding_dim=self.latent_dim)
        self.user_feature_embeddings = torch.nn.Embedding(num_embeddings=self.user_feature_num, embedding_dim=self.latent_dim)
        self.w_linears = nn.ModuleList()
        self.w_linears.append(nn.Linear(self.latent_dim*2, self.latent_dim*1)) #TODO: change
        for i in range(self.num_layers-1):
            self.w_linears.append(nn.Linear(self.latent_dim*2, self.latent_dim*1))
        self.v_linears = nn.ModuleList([nn.Linear(self.latent_dim*1, self.latent_dim*1) for _ in range(self.num_layers)])
        self.optim = optim.Adam(self.parameters(), lr=config['lr'])
        self.user_proj = torch.nn.Linear(self.latent_dim*2+300, self.latent_dim)
        self.item_proj = torch.nn.Linear(self.latent_dim*2+300, self.latent_dim)
        self.device = self.config['device']
        self.test_item_emb = None
        with open(f'./data/text/product_name_count{suffix}.pkl', 'rb') as f:
            self.item_name = pickle.load(f)
        with open(f'./data/text/product_main_comment_count{suffix}.pkl', 'rb') as f:
            self.item_main_comment = pickle.load(f)
        with open(f'./data/text/product_main_list_comment_count{suffix}.pkl', 'rb') as f:
            self.item_main_list_comment = pickle.load(f)
            
        with open(f'./data/text/user_name_count{suffix}.pkl', 'rb') as f:
            self.user_name = pickle.load(f)
        with open(f'./data/text/user_main_comment_count{suffix}.pkl', 'rb') as f:
            self.user_main_comment = pickle.load(f)
        with open(f'./data/text/user_main_list_comment_count{suffix}.pkl', 'rb') as f:
            self.user_main_list_comment = pickle.load(f)
        #self.item_name=torch.from_numpy(np.load(f'./data/text/product_name_count{suffix}.npy'))
        #self.item_main_comment=torch.from_numpy(np.load(f'./data/text/product_main_comment_count{suffix}.npy'))
        #self.item_main_list_comment=torch.from_numpy(np.load(f'./data/text/product_main_list_comment_count{suffix}.npy'))
        #self.user_name_count = torch.from_numpy(np.load(f'./data/text/user_name_count{suffix}.npy'))
        #self.user_main_comment=torch.from_numpy(np.load(f'./data/text/user_main_comment_count{suffix}.npy'))
        #self.user_main_list_comment=torch.from_numpy(np.load(f'./data/text/user_main_list_comment_count{suffix}.npy'))
        print('loaded text vecs')
        self.vocab_num = self.item_name.shape[1]
        self.word_emb_dim = self.latent_dim//2
        self.word_embedding = torch.nn.Embedding(num_embeddings=self.item_name.shape[1], embedding_dim=self.word_emb_dim)
        self.init_parameters()
        
    def init_parameters(self):
        gain = nn.init.calculate_gain('relu')
        gain=0.1
        nn.init.xavier_uniform_(self.item_feature_embeddings.weight, gain=gain)
        nn.init.xavier_uniform_(self.user_feature_embeddings.weight, gain=gain)
        nn.init.xavier_uniform_(self.word_embedding.weight, gain=gain)
        nn.init.normal_(self.user_id_embeddings.weight, std=gain)
        nn.init.normal_(self.item_id_embeddings.weight, std=gain)
        for i, w in enumerate(self.w_linears):
            if i == self.num_layers - 1:
                nn.init.xavier_uniform_(w.weight)
            else:
                nn.init.xavier_uniform_(w.weight, gain=gain)
            nn.init.zeros_(w.bias)
            
    def get_text_embedding(self, index, mode='user'):
        if mode=='user':
            name = self.user_name[index.tolist()].tocoo()
            main_comment = self.user_main_list_comment[index.tolist()].tocoo()
        elif mode=='item':
            name = self.item_name[index.tolist()].tocoo()
            main_comment = self.item_main_list_comment[index.tolist()].tocoo()
            
        #name_count_source, indptr = name_count.indices, name_count.indptr
        #name_source_word_embedding = self.word_embedding(torch.from_numpy(name_count_source).to(self.device))
        #name_out = torch.zeros((len(index), self.word_emb_dim)).to(self.device)
        #for i in range(len(index)):
        #    name_out[i] = torch.mean(name_source_word_embedding[indptr[i]:indptr[i+1]], dim=0)
        #name_out = torch.zeros((index.size(0), self.latent_dim//2)).to(self.device)
        #name_embedding = scatter(name_source_word_embedding, name_count_target, out=name_out, dim=0, reduce='mean')
        name_source, name_target= name.col, name.row
        #print(name_source)
        name_source_word_embedding = self.word_embedding(torch.from_numpy(name_source).to(self.device))
        name_out = torch.zeros((len(index), self.word_emb_dim)).to(self.device)
        #print(name_source_word_embedding, torch.from_numpy(name_target).to(self.device), name_out)
        name_out = scatter(name_source_word_embedding, torch.from_numpy(name_target).long().to(self.device), dim=0, out=name_out, reduce='mean')
        
        comment_source, comment_target = main_comment.col, main_comment.row
        comment_source_word_embedding = self.word_embedding(torch.from_numpy(comment_source).to(self.device))
        comment_out = torch.zeros((len(index), self.word_emb_dim)).to(self.device)
        comment_out = scatter(comment_source_word_embedding, torch.from_numpy(comment_target).long().to(self.device), dim=0, out=comment_out, reduce='mean')
        #comment_count_source, indptr = main_comment_count.indices, main_comment_count.indptr
        #comment_source_word_embedding = self.word_embedding(torch.from_numpy(comment_count_source).to(self.device))
        #comment_out = torch.zeros((len(index), self.latent_dim//2)).to(self.device)
        #comment_embedding = scatter(comment_source_word_embedding, comment_count_target, out=comment_out, dim=0, reduce='mean') 
        #comment_out = torch.zeros((len(index), self.word_emb_dim)).to(self.device)
        #for i in range(len(index)):
            #comment_out[i] = torch.mean(comment_source_word_embedding[indptr[i]:indptr[i+1]], dim=0)
        #print(comment_out.size())
        text_embedding = torch.cat([name_out, comment_out], dim=1)
        #text_embedding = torch.cat([name_embedding, comment_embedding], dim=1)
        return text_embedding
            
    
    def get_initial_user_emb(self, user):
        id_embedding = self.user_id_embeddings(user.to(self.device))
        text_embedding = self.get_text_embedding(user, mode='user')
        word_embedding = self.user_word_embedding[user].to(self.device)
        feature_embedding = torch.mean(self.user_feature_embeddings(self.user_features[user].to(self.device)), dim=1)
        feature_embedding = torch.cat([text_embedding, word_embedding, feature_embedding], dim=1)
        feature_embedding = self.user_proj(feature_embedding)
        user_embedding = torch.cat([id_embedding, feature_embedding], dim=1)
        return feature_embedding
    
    def get_initial_item_emb(self, item):
        id_embedding = self.item_id_embeddings(item.to(self.device))
        text_embedding = self.get_text_embedding(item, mode='item')
        word_embedding = self.item_word_embedding[item].to(self.device)
        feature_embedding = torch.mean(self.item_feature_embeddings(self.item_features[item].to(self.device)), dim=1)
        feature_embedding = torch.cat([text_embedding, word_embedding, feature_embedding], dim=1)
        feature_embedding = self.item_proj(feature_embedding)
        item_embedding = torch.cat([id_embedding, feature_embedding], dim=1)
        return feature_embedding
    
    def get_initial_emb(self, index):
        user_index, item_index = (index<self.n_user).to(self.device), (index>=self.n_user).to(self.device)
        user, item = index[user_index], index[item_index]
        emb = torch.zeros((len(index), self.latent_dim*1)).to(self.device) #TODO: change
        #print(user, item)
        emb[user_index] = self.get_initial_user_emb(user)
        emb[item_index] = self.get_initial_item_emb(item-self.n_user)
        return emb
        
    def forward(self, x, adjs):
        #neighbors: [batch(item), batch*neighbor_size(user), batch*neighbor_size^2(item)]
        #          or [batch(user), batch*neighbor_size(item), batch*neighbor_size^2(user)]
        #offsets: [batch, batch*neighbor_size, batch*neighbor_size*neighbor_size]
        for i, (edge_index, _, size) in enumerate(adjs):
            source_x = x[edge_index[0]]
            source_x = self.dropout(source_x)
            target_x = x[:size[1]]
            out = torch.zeros((size[1], source_x.size(1))).to(self.device)
            aggr = scatter(source_x, edge_index[1], dim=0, out=out, reduce='mean')
            x = self.w_linears[i](torch.cat([target_x, aggr], dim=1))
            if i!=self.num_layers-1:
                x = x.relu()
        return x

            
    def loss(self, user_emb: torch.Tensor, pos_emb: torch.Tensor, neg_emb:torch.Tensor):
        pos_scores = torch.sum(torch.mul(user_emb, pos_emb), dim=1)
        neg_scores = torch.sum(torch.mul(user_emb, neg_emb), dim=1)
        all_param = 0
        for p in self.parameters():
            all_param+=all_param+p.norm(2)
        all_param = all_param/user_emb.size(0)
        loss = torch.mean(torch.nn.functional.softplus(neg_scores-pos_scores))
        loss = loss+all_param*self.config['decay']
        return loss
        
    
    def OneEpoch(self, user, pos, neg):
        pos = pos + self.n_user
        neg = neg + self.n_user
        #print(self.edge_index)
        #print(user, pos, neg)
        user_loader = NeighborSampler(self.edge_index, node_idx=user, sizes=[self.num_neighbors for _ in range(self.num_layers)],
                                      batch_size=self.config['bpr_batch_size'], shuffle=False, num_workers=8)
        pos_loader = NeighborSampler(self.edge_index, node_idx=pos, sizes=[self.num_neighbors for _ in range(self.num_layers)],
                                batch_size=self.config['bpr_batch_size'], shuffle=False, num_workers=8)
        neg_loader = NeighborSampler(self.edge_index, node_idx=neg, sizes=[self.num_neighbors, self.num_neighbors],
                                batch_size=self.config['bpr_batch_size'], shuffle=False, num_workers=8)
        total_batch = len(user)//self.config['bpr_batch_size'] + 1
        aver_loss = 0
        for (user_batch_size, user_id, user_adjs), (pos_batch_size, pos_id, pos_adjs), (neg_batch_size, neg_id, neg_adjs) in tqdm(zip(user_loader, pos_loader, neg_loader)):
            assert user_batch_size == pos_batch_size == neg_batch_size
            #user_x, pos_x, neg_x = self.get_initial_train_emb(user_id, pos_id, neg_id)
            
            user_x = self.get_initial_emb(user_id)
            pos_x, neg_x = self.get_initial_emb(pos_id), self.get_initial_emb(neg_id)
            
            user_adjs = [user_adj.to(self.device) for user_adj in user_adjs]
            pos_adjs = [pos_adj.to(self.device) for pos_adj in pos_adjs]
            neg_adjs = [neg_adj.to(self.device) for neg_adj in neg_adjs]
            user_emb = self.forward(user_x, user_adjs)
            pos_emb = self.forward(pos_x, pos_adjs)
            neg_emb = self.forward(neg_x,neg_adjs)
            self.optim.zero_grad()
            loss = self.loss(user_emb, pos_emb, neg_emb)
            aver_loss+=loss.detach().cpu()
            loss.backward()
            #loss.backward(retain_graph=True)
            del loss
            self.optim.step()
        aver_loss/=total_batch
        return aver_loss
    
    
        
    @torch.no_grad()
    def getUsersRating(self, users):
        trainItem = torch.tensor(self.dataset.trainItem).to(self.device)
        trainUser =  torch.tensor(self.dataset.trainUser).to(self.device)
        user_initial_emb = self.get_initial_user_emb(users)
        item_initial_emb = self.get_initial_item_emb(torch.arange(self.m_item))
        
        user_x, item_x = user_initial_emb, item_initial_emb
        for i in range(self.num_layers):
            user_out, item_out = torch.zeros_like(user_x), torch.zeros_like(item_x)
            user_x_all, item_x_all = user_x[trainUser], item_x[trainItem]
            user_x_agg = scatter(item_x_all, trainUser, out=user_out, dim=0, reduce='mean')
            item_x_agg = scatter(user_x_all, trainItem, out=item_out, dim=0, reduce='mean')
            user_x = self.w_linears[i](torch.cat([user_x, user_x_agg], dim=1))
            item_x = self.w_linears[i](torch.cat([item_x, item_x_agg], dim=1))
            if i!=self.num_layers-1:
                item_x = item_x.relu()
                user_x = user_x.relu()

        rating = torch.matmul(user_x[users], item_x.T)
        return rating 
        
            
    
        
        
    
    
        
        
        
        
        
        
    
        
        
        
        
        