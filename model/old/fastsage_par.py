import itertools
import operator
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.multiprocessing import Manager
from torch_scatter.scatter import scatter
from tqdm import tqdm

from neighbor_sampling import UniformNeighbors, uniform_neighbors
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


class FastSAGEPar(torch.nn.Module):
    def __init__(self, config, dataset):
        super().__init__()
        self.dataset = dataset
        self.config = config
        self.latent_dim = self.config["recdim"]
        self.num_layers = self.config["layer"]
        self.num_neighbors = self.config["num_neighbors"]
        self.dropout = nn.Dropout(0.2)

        self.user_features = np.load(
            "/home/yamanishi/project/furusato_recommend/data/cb/customer_features.npy",
            allow_pickle=True,
        )
        self.item_features = np.load(
            "/home/yamanishi/project/furusato_recommend/data/cb/product_features.npy",
            allow_pickle=True,
        )
        self.user_feature_indices, self.user_offsets = get_indice_offset(
            self.user_features
        )
        self.item_feature_indices, self.item_offsets = get_indice_offset(
            self.item_features
        )
        # self.user_feature_num = max([max(ufe) for ufe in self.user_features])
        # self.item_feature_num = max([max(ife) for ife in self.item_features])
        self.user_feature_num = 3207
        self.item_feature_num = 2094
        # print(self.user_feature_num)
        # print(self.item_feature_num)
        self.item_feature_embeddings = torch.nn.Embedding(
            num_embeddings=self.item_feature_num, embedding_dim=self.latent_dim
        )
        self.user_feature_embeddings = torch.nn.Embedding(
            num_embeddings=self.user_feature_num, embedding_dim=self.latent_dim
        )
        self.w_linears = nn.ModuleList(
            [
                nn.Linear(self.latent_dim * 2, self.latent_dim)
                for _ in range(self.num_layers)
            ]
        )
        self.init_parameters()
        self.optim = optim.Adam(self.parameters(), lr=config["lr"])
        self.user_proj = torch.nn.Linear(self.latent_dim, self.latent_dim)
        self.item_proj = torch.nn.Linear(self.latent_dim, self.latent_dim)
        self.device = self.config["device"]
        self.process_num = 16
        self.neighbor_sampler = UniformNeighbors(
            dataset, self.num_layers, self.num_neighbors
        )

    def init_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.item_feature_embeddings.weight)
        nn.init.xavier_uniform_(self.user_feature_embeddings.weight)
        for i, w in enumerate(self.w_linears):
            if i == self.num_layers - 1:
                nn.init.xavier_uniform_(w.weight)
            else:
                nn.init.xavier_uniform_(w.weight, gain=gain)
            nn.init.zeros_(w.bias)

    def get_initial_embedding(self):
        user_initial_embedding = self.user_proj(
            F.embedding_bag(
                torch.tensor(self.user_feature_indices).to(self.device),
                self.user_feature_embeddings.weight,
                offsets=torch.tensor(self.user_offsets).to(self.device),
                mode="mean",
            )
        )
        item_initial_embedding = self.item_proj(
            F.embedding_bag(
                torch.tensor(self.item_feature_indices).to(self.device),
                self.item_feature_embeddings.weight,
                offsets=torch.tensor(self.item_offsets).to(self.device),
                mode="mean",
            )
        )

        return user_initial_embedding, item_initial_embedding

    def get_initial_hiddens(self, neighbors, mode):
        initial_hiddens = []
        user_initial_embedding = self.user_proj(
            F.embedding_bag(
                torch.tensor(self.user_feature_indices).to(self.device),
                self.user_feature_embeddings.weight,
                offsets=torch.tensor(self.user_offsets).to(self.device),
                mode="mean",
            )
        )
        item_initial_embedding = self.item_proj(
            F.embedding_bag(
                torch.tensor(self.item_feature_indices).to(self.device),
                self.item_feature_embeddings.weight,
                offsets=torch.tensor(self.item_offsets).to(self.device),
                mode="mean",
            )
        )
        offset = 0 if mode == "user" else 1
        for i in range(len(neighbors)):
            if offset % 2:
                initial_hiddens.append(
                    item_initial_embedding[torch.tensor(neighbors[i]).to(self.device)]
                )
            else:
                initial_hiddens.append(
                    user_initial_embedding[torch.tensor(neighbors[i]).to(self.device)]
                )
            offset += 1
        return initial_hiddens

    def get_layer_indice_offset(self, neighbors, offsets):
        len_neighbors = sum([len(n) for n in neighbors])
        source = torch.arange(len_neighbors)
        begin_node_offset = 0
        batch_size = len(offsets[0])
        neighbor_num = self.num_neighbors
        target = torch.full((batch_size,), 0)
        layer_offsets = deque([])
        for layer in range(self.num_layers):
            target_tmp = torch.arange(
                begin_node_offset,
                begin_node_offset + batch_size * (neighbor_num**layer),
            )
            target_tmp = torch.repeat_interleave(target_tmp, neighbor_num)
            begin_node_offset += batch_size * (neighbor_num**layer)
            target = torch.cat([target, target_tmp])
            layer_offsets.appendleft(begin_node_offset)
        layer_offsets.appendleft(len_neighbors)
        return source.long(), target.long(), layer_offsets

    def forward(self, neighbors, offsets, mode):
        # neighbors: [batch(item), batch*neighbor_size(user), batch*neighbor_size^2(item)]
        #          or [batch(user), batch*neighbor_size(item), batch*neighbor_size^2(user)]
        # offsets: [batch, batch*neighbor_size, batch*neighbor_size*neighbor_size]
        source, target, layer_offsets = self.get_layer_indice_offset(neighbors, offsets)
        hidden = self.get_initial_hiddens(neighbors, mode=mode)
        batch_size = hidden[0].size(0)
        hidden = torch.cat(hidden, dim=0)
        x = hidden

        for i in range(self.num_layers):
            source_layer = source[: layer_offsets[i]]
            target_layer = target[: layer_offsets[i]]
            source_x = x[source_layer]
            source_x[:batch_size] = 0
            out = torch.zeros((layer_offsets[i + 1], self.latent_dim))
            aggregated = scatter(
                source_x, target_layer.to(self.device), out=out.to(self.device), dim=0
            )
            x = self.w_linears[i](
                torch.cat([x[: layer_offsets[i + 1]], aggregated], dim=1)
            )
            # x = scatter(source_x, target_layer, out=out, dim=1)
        return x

    def loss(
        self, user_emb: torch.Tensor, pos_emb: torch.Tensor, neg_emb: torch.Tensor
    ):
        pos_scores = torch.sum(torch.mul(user_emb, pos_emb), dim=1)
        neg_scores = torch.sum(torch.mul(user_emb, neg_emb), dim=1)
        all_param = 0
        for p in self.parameters():
            all_param += all_param + p.norm(2).pow(2)
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        loss = loss + all_param * self.config["decay"]
        return loss

    def stageOne(self, user: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor):
        # if torch.multiprocessing.get_start_method() == 'fork':
        #    torch.multiprocessing.set_start_method('spawn', force=True)
        user, pos, neg = list(user.cpu()), list(pos.cpu()), list(neg.cpu())
        start = time.time()
        result = self.neighbor_sampler.sample(user, pos, neg)
        # print('spent seconds for neighbor sample', time.time()-start)
        total_loss = 0
        for (
            user_neighbors,
            user_offsets,
            pos_neighbors,
            pos_offsets,
            neg_neighbors,
            neg_offsets,
        ) in result:
            # print(user_neighbors)
            user_emb = self.forward(user_neighbors, user_offsets, mode="user")
            pos_emb = self.forward(pos_neighbors, pos_offsets, mode="item")
            neg_emb = self.forward(neg_neighbors, neg_offsets, mode="item")
            self.optim.zero_grad()
            loss = self.loss(user_emb, pos_emb, neg_emb)
            total_loss += loss
            loss.backward()
            self.optim.step()
        return total_loss / self.process_num

    @torch.no_grad()
    def getUsersRating(self, users):
        trainItem = torch.tensor(self.dataset.trainItem).to(self.device)
        trainUser = torch.tensor(self.dataset.trainUser).to(self.device)
        user_intitial_emb, item_initial_emb = self.get_initial_embedding()
        user_x, item_x = user_intitial_emb, item_initial_emb
        for i in range(self.num_layers):
            user_out, item_out = torch.zeros_like(user_x), torch.zeros_like(item_x)
            user_x_all, item_x_all = user_x[trainUser], item_x[trainItem]
            user_x_agg = scatter(item_x_all, trainUser, out=user_out, dim=0)
            item_x_agg = scatter(user_x_all, trainItem, out=item_out, dim=0)
            user_x = self.w_linears[i](torch.cat([user_x, user_x_agg], dim=1))
            item_x = self.w_linears[i](torch.cat([item_x, item_x_agg], dim=1))
            if i != self.num_layers - 1:
                item_x = item_x.relu()
                user_x = user_x.relu()

        rating = torch.matmul(user_x[users], item_x.T)
        return rating
