import torch
import torch.nn as nn
from torch_geometric.nn.conv import LGConv
import world
import torch.optim as optim
from torch_scatter.scatter import scatter
from tqdm import tqdm
from utils import minibatch

class rAdjConv(torch.nn.Module):
    def __init__(self, config, dataset):
        super().__init__()
        self.dataset = dataset
        self.config = config
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.train_user = torch.tensor(dataset.trainUser)
        self.train_item = torch.tensor(dataset.trainItem) + self.num_users
        self.train_edge = torch.cat([torch.stack([self.train_user, self.train_item]), torch.stack([self.train_item, self.train_user])], dim=1).to(config['device'])        
        self.all_div = torch.zeros(self.num_users+self.num_items)
        value, count = torch.unique(self.train_edge[0].cpu(), return_counts=True)
        self.all_div[value.long()] = count.float()
        self.all_div[self.all_div==0] = 1e-6
        self.r = self.config['r']
        self.train_edge_div = ((self.all_div[self.train_edge[0]]**self.r)*(self.all_div[self.train_edge[1]]**(1-self.r))).to(config['device'])
        self.device = self.config['device']
        
    def forward(self, x):
        device = x.device
        out = torch.zeros_like(x).to(device)
        x = x[self.train_edge[0]]/self.train_edge_div.unsqueeze(1)
        out = scatter(x, self.train_edge[1], out=out, dim=0)
        return out
        

class rAdjGCN(torch.nn.Module):
    def __init__(self, config, dataset):
        super().__init__()
        self.dataset = dataset
        self.config = config
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config['recdim']
        self.num_layers = self.config['layer']
        self.train_user = torch.tensor(dataset.trainUser)
        self.train_item = torch.tensor(dataset.trainItem) + self.num_users
        self.train_edge = torch.cat([torch.stack([self.train_user, self.train_item]), torch.stack([self.train_item, self.train_user])], dim=1).to(config['device'])        
        self.__init_weight()
        self.optim = optim.Adam(self.parameters(), lr=config['lr'])
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(rAdjConv(config, dataset))
            
        self.device = self.config['device']
        
    def __init_weight(self):
        self.all_embedding = torch.nn.Embedding(num_embeddings=self.num_items+self.num_users, embedding_dim=self.latent_dim)
        nn.init.normal_(self.all_embedding.weight, std=0.1)
        world.cprint('use NORMAL distribution initilizer')
        
    def forward(self):
        x = self.all_embedding.weight
        x_out = x
        for i in range(self.num_layers):
            x = self.layers[i](x)
            x_out = x_out+x
        x_out = x_out/(1+self.num_layers)
        user_out, item_out = x_out[:self.num_users], x_out[self.num_users:]
        return user_out, item_out
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.forward()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.all_embedding(users)
        pos_emb_ego = self.all_embedding(pos_items+self.num_users)
        neg_emb_ego = self.all_embedding(neg_items+self.num_users)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss
    
    def getUsersRating(self, users):
        all_users, all_items = self.forward()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = torch.matmul(users_emb, items_emb.t())
        return rating
    
    def stageOne(self, user, pos, neg):
        self.optim.zero_grad()
        loss, reg_loss = self.bpr_loss(user, pos, neg)
        loss = loss + self.config['decay']*reg_loss
        loss.backward()
        self.optim.step()
        return loss
    
    def OneEpoch(self, user, pos, neg):
        aver_loss = 0
        total_batch = len(user)//self.config['bpr_batch_size'] + 1
        for batch_i, (u, posItem, negItem) in tqdm(enumerate(minibatch(user, pos, neg, batch_size=self.config['bpr_batch_size']))):
            u, posItem, negItem = u.to(self.device), posItem.to(self.device), negItem.to(self.device)
            loss = self.stageOne(u, posItem, negItem)
            aver_loss+=loss
        aver_loss/=total_batch
        return aver_loss

        
        