import itertools
import operator
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


class GraphSAGE(torch.nn.Module):
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

        self.user_features = np.load(
            f"/home/yamanishi/project/furusato_recommend/data/cb/customer_feature_pad{suffix}.npy",
            allow_pickle=True,
        )
        self.item_features = np.load(
            f"/home/yamanishi/project/furusato_recommend/data/cb/product_feature_pad{suffix}.npy",
            allow_pickle=True,
        )
        self.item_text_features = (
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
        # self.user_feature_num = 3207
        # self.item_feature_num = 2094
        # print(self.user_feature_num)
        # print(self.item_feature_num)
        self.user_id_embeddings = torch.nn.Embedding(
            num_embeddings=self.n_user, embedding_dim=self.latent_dim
        )
        self.item_id_embeddings = torch.nn.Embedding(
            num_embeddings=self.m_item, embedding_dim=self.latent_dim
        )
        self.item_feature_embeddings = torch.nn.Embedding(
            num_embeddings=self.item_feature_num, embedding_dim=self.latent_dim
        )
        self.user_feature_embeddings = torch.nn.Embedding(
            num_embeddings=self.user_feature_num, embedding_dim=self.latent_dim
        )
        self.w_linears = nn.ModuleList()
        self.w_linears.append(nn.Linear(self.latent_dim * 4, self.latent_dim * 1))
        for i in range(self.num_layers - 1):
            self.w_linears.append(nn.Linear(self.latent_dim * 2, self.latent_dim * 1))
        self.v_linears = nn.ModuleList(
            [
                nn.Linear(self.latent_dim * 1, self.latent_dim * 1)
                for _ in range(self.num_layers)
            ]
        )
        self.init_parameters()
        self.optim = optim.Adam(self.parameters(), lr=config["lr"])
        self.user_proj = torch.nn.Linear(self.latent_dim, self.latent_dim)
        self.item_proj = torch.nn.Linear(self.latent_dim + 300, self.latent_dim)
        self.device = self.config["device"]
        self.test_item_emb = None
        self.G1 = torch.nn.Linear(self.latent_dim, self.latent_dim)
        self.G2 = torch.nn.Linear(self.latent_dim, self.latent_dim, bias=False)

    def init_parameters(self):
        gain = nn.init.calculate_gain("relu")
        gain = 0.1
        nn.init.xavier_uniform_(self.item_feature_embeddings.weight, gain=gain)
        nn.init.xavier_uniform_(self.user_feature_embeddings.weight, gain=gain)
        nn.init.normal_(self.user_id_embeddings.weight, std=gain)
        nn.init.normal_(self.item_id_embeddings.weight, std=gain)
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
        item_features = F.embedding_bag(
            torch.tensor(self.item_feature_indices).to(self.device),
            self.item_feature_embeddings.weight,
            offsets=torch.tensor(self.item_offsets).to(self.device),
            mode="mean",
        )
        item_text_features = self.item_text_features.to(self.device)
        user_initial_embedding = torch.cat(
            [self.user_id_embeddings.weight, user_initial_embedding], dim=1
        )
        item_initial_embedding = self.item_proj(
            torch.cat([item_features, item_text_features], dim=1)
        )
        item_initial_embedding = torch.cat(
            [self.item_id_embeddings.weight, item_initial_embedding], dim=1
        )

        return user_initial_embedding, item_initial_embedding

    def get_initial_train_emb(self, user, pos, neg):
        # print(torch.from_numpy(self.user_features[user]))
        user_user_index, user_item_index = (user < self.n_user).to(self.device), (
            user >= self.n_user
        ).to(self.device)
        pos_user_index, pos_item_index = (pos < self.n_user).to(self.device), (
            pos >= self.n_user
        ).to(self.device)
        neg_user_index, neg_item_index = (neg < self.n_user).to(self.device), (
            neg >= self.n_user
        ).to(self.device)
        user_user, user_item = (
            user[user_user_index],
            user[user_item_index] - self.n_user,
        )
        pos_user, pos_item = pos[pos_user_index], pos[pos_item_index] - self.n_user
        neg_user, neg_item = neg[neg_user_index], neg[neg_item_index] - self.n_user
        user_x, pos_x, neg_x = (
            torch.zeros((user.size(0), self.latent_dim * 2)).to(self.device),
            torch.zeros((pos.size(0), self.latent_dim * 2)).to(self.device),
            torch.zeros((neg.size(0), self.latent_dim * 2)).to(self.device),
        )
        user_x[user_user_index] = torch.cat(
            [
                self.user_id_embeddings(user_user.to(self.device)),
                self.user_proj(
                    torch.mean(
                        self.user_feature_embeddings(
                            torch.from_numpy(self.user_features[user_user]).to(
                                self.device
                            )
                        ),
                        dim=1,
                    )
                ),
            ],
            dim=1,
        )
        user_x[user_item_index] = torch.cat(
            [
                self.item_id_embeddings(user_item.to(self.device)),
                self.item_proj(
                    torch.cat(
                        [
                            torch.mean(
                                self.item_feature_embeddings(
                                    torch.from_numpy(self.item_features[user_item]).to(
                                        self.device
                                    )
                                ),
                                dim=1,
                            ),
                            self.item_text_features[user_item],
                        ],
                        dim=1,
                    )
                ),
            ],
            dim=1,
        )
        pos_x[pos_user_index] = torch.cat(
            [
                self.user_id_embeddings(pos_user.to(self.device)),
                self.user_proj(
                    torch.mean(
                        self.user_feature_embeddings(
                            torch.from_numpy(self.user_features[pos_user]).to(
                                self.device
                            )
                        ),
                        dim=1,
                    )
                ),
            ],
            dim=1,
        )
        pos_x[pos_item_index] = torch.cat(
            [
                self.item_id_embeddings(pos_item.to(self.device)),
                self.item_proj(
                    torch.cat(
                        [
                            torch.mean(
                                self.item_feature_embeddings(
                                    torch.from_numpy(self.item_features[pos_item]).to(
                                        self.device
                                    )
                                ),
                                dim=1,
                            ),
                            self.item_text_features[pos_item],
                        ],
                        dim=1,
                    )
                ),
            ],
            dim=1,
        )
        neg_x[neg_user_index] = torch.cat(
            [
                self.user_id_embeddings(neg_user.to(self.device)),
                self.user_proj(
                    torch.mean(
                        self.user_feature_embeddings(
                            torch.from_numpy(self.user_features[neg_user]).to(
                                self.device
                            )
                        ),
                        dim=1,
                    )
                ),
            ],
            dim=1,
        )
        neg_x[neg_item_index] = torch.cat(
            [
                self.item_id_embeddings(neg_item.to(self.device)),
                self.item_proj(
                    torch.cat(
                        [
                            torch.mean(
                                self.item_feature_embeddings(
                                    torch.from_numpy(self.item_features[neg_item]).to(
                                        self.device
                                    )
                                ),
                                dim=1,
                            ),
                            self.item_text_features[neg_item],
                        ],
                        dim=1,
                    )
                ),
            ],
            dim=1,
        )
        return user_x, pos_x, neg_x

    def get_text_embedding(self, index, mode="user"):
        return 0

    def get_id_embedding(self, index, mode="user"):
        if mode == "user":
            return self.user_id_embeddings(index.to(self.device))

        elif mode == "item":
            return self.item_id_embedding(index.to(self.device))

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
        pos = pos + self.n_user
        neg = neg + self.n_user
        user_loader = NeighborSampler(
            self.edge_index,
            node_idx=user,
            sizes=[self.num_neighbors for _ in range(self.num_layers)],
            batch_size=self.config["bpr_batch_size"],
            shuffle=False,
            num_workers=8,
        )
        pos_loader = NeighborSampler(
            self.edge_index,
            node_idx=pos,
            sizes=[self.num_neighbors for _ in range(self.num_layers)],
            batch_size=self.config["bpr_batch_size"],
            shuffle=False,
            num_workers=8,
        )
        neg_loader = NeighborSampler(
            self.edge_index,
            node_idx=neg,
            sizes=[self.num_neighbors, self.num_neighbors],
            batch_size=self.config["bpr_batch_size"],
            shuffle=False,
            num_workers=8,
        )
        total_batch = len(user) // self.config["bpr_batch_size"] + 1
        aver_loss = 0
        user_init_x, item_init_x = self.get_initial_embedding()
        x = torch.cat([user_init_x, item_init_x], dim=0)
        for (
            (user_batch_size, user_id, user_adjs),
            (pos_batch_size, pos_id, pos_adjs),
            (neg_batch_size, neg_id, neg_adjs),
        ) in tqdm(zip(user_loader, pos_loader, neg_loader)):
            assert user_batch_size == pos_batch_size == neg_batch_size
            if self.config["train_emb"]:
                user_x, pos_x, neg_x = self.get_initial_train_emb(
                    user_id, pos_id, neg_id
                )
            else:
                user_x, pos_x, neg_x = x[user_id], x[pos_id], x[neg_id]
            user_adjs = [user_adj.to(self.device) for user_adj in user_adjs]
            pos_adjs = [pos_adj.to(self.device) for pos_adj in pos_adjs]
            neg_adjs = [neg_adj.to(self.device) for neg_adj in neg_adjs]
            user_emb = self.forward(user_x, user_adjs)
            pos_emb = self.forward(pos_x, pos_adjs)
            neg_emb = self.forward(neg_x, neg_adjs)
            self.optim.zero_grad()
            loss = self.loss(user_emb, pos_emb, neg_emb)
            aver_loss += loss.detach().cpu()
            if self.config["train_emb"]:
                loss.backward()
            else:
                loss.backward(retain_graph=True)
            # loss.backward(retain_graph=True)
            del loss
            self.optim.step()
        aver_loss /= total_batch
        return aver_loss

    @torch.no_grad()
    def getUsersRating(self, users):
        if self.config["inference"] == "all":
            trainItem = torch.tensor(self.dataset.trainItem).to(self.device)
            trainUser = torch.tensor(self.dataset.trainUser).to(self.device)
            user_intitial_emb, item_initial_emb = self.get_initial_embedding()
            user_x, item_x = user_intitial_emb, item_initial_emb
            for i in range(self.num_layers):
                user_out, item_out = torch.zeros_like(user_x), torch.zeros_like(item_x)
                user_x_all, item_x_all = user_x[trainUser], item_x[trainItem]
                user_x_agg = scatter(
                    item_x_all, trainUser, out=user_out, dim=0, reduce="mean"
                )
                item_x_agg = scatter(
                    user_x_all, trainItem, out=item_out, dim=0, reduce="mean"
                )
                user_x = self.w_linears[i](torch.cat([user_x, user_x_agg], dim=1))
                item_x = self.w_linears[i](torch.cat([item_x, item_x_agg], dim=1))
                if i != self.num_layers - 1:
                    item_x = item_x.relu()
                    user_x = user_x.relu()

            rating = torch.matmul(user_x[users], item_x.T)
            return rating

        elif self.config["inference"] == "sample":
            all_item = torch.arange(self.n_user, self.n_user + self.m_item)
            user_initial_emb, item_initial_emb = self.get_initial_embedding()
            x = torch.cat([user_initial_emb, item_initial_emb], dim=0)
            item_emb_all = []
            if self.test_item_emb == None:
                print("sample item first")
                item_sampler = NeighborSampler(
                    self.edge_index,
                    node_idx=all_item,
                    sizes=[self.num_neighbors for _ in range(self.num_layers)],
                    batch_size=self.config["test_u_batch_size"],
                    shuffle=False,
                    num_workers=8,
                )
                for item_batch_size, item_id, item_adjs in item_sampler:
                    adj = [adj.to(self.device) for adj in item_adjs]
                    item_emb = self.forward(x[item_id], adj)
                    item_emb_all.append(item_emb)
                self.test_item_emb = torch.cat(item_emb_all, dim=0)

            user_sampler = NeighborSampler(
                self.edge_index,
                node_idx=torch.tensor(users),
                sizes=[self.num_neighbors * 5 for _ in range(self.num_layers)],
                batch_size=self.config["test_u_batch_size"],
                shuffle=False,
                num_workers=8,
            )
            rating_list, groundTrue_list = [], []
            users_list = []
            for user_batch_size, user_id, user_adjs in tqdm(user_sampler):
                exclude_index = []
                exclude_items = []
                allPos = self.dataset.getUserPosItems(user_id[:user_batch_size])
                for range_i, items in enumerate(allPos):
                    exclude_index.extend([range_i] * len(items))
                    exclude_items.extend(items)
                adj = [adj.to(self.device) for adj in user_adjs]
                user_emb = self.forward(x[user_id], adj)
                rating = torch.matmul(user_emb, self.test_item_emb.T)
                rating = rating.cpu()

                _, rating_K = torch.topk(rating, k=max(world.topks))
                del rating
                rating_list.append(rating_K.cpu())
                groundTrue_list.append(
                    [self.dataset.testDict[u.item()] for u in user_id[:user_batch_size]]
                )
                users_list.append(list(user_id[:user_batch_size]))
                # user_emb_all.append(user_emb)
            return users_list, groundTrue_list, rating_list
