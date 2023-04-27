import itertools
import operator
import os
import pickle
import random
import sys
from collections import deque, defaultdict
from os.path import join
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.sparse import csr_matrix
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch_geometric.loader import NeighborSampler
from torch_scatter.scatter import scatter
from tqdm import tqdm
from utils import join_list

import wandb
from metric import (Coverage, Diversity, Metric, NDCGatK_r, Novelty,
                    RecallPrecision_ATk, Unexpectedness)

NEGATIVE_POW=0.2
POSITIVE_NUM_LIMIT=3000
TRAIN_ITERATIVE=3
TEST_COUNT=100
TEST_SPAN=10

class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")
    
    @property
    def n_users(self):
        raise NotImplementedError
    
    @property
    def m_items(self):
        raise NotImplementedError
    
    @property
    def trainDataSize(self):
        raise NotImplementedError
    
    @property
    def testDict(self):
        raise NotImplementedError
    
    @property
    def allPos(self):
        raise NotImplementedError
    
    def getUserItemFeedback(self, users, items):
        raise NotImplementedError
    
    def getUserPosItems(self, users):
        raise NotImplementedError
    
    def getUserNegItems(self, users):
        """
        not necessary for large dataset
        it's stupid to return all neg items in super large dataset
        """
        raise NotImplementedError
    
    def getSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A = 
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError

class Loader(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    gowalla dataset
    """

    def __init__(self,config,path="./data/cf"):
        # train or test
        self.split = config['A_split']
        self.folds = config['A_n_fold']
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        self.n_user = 0
        self.m_item = 0
        suffix = config['suffix']
        train_file = path + f'/train{suffix}.txt'
        test_file = path + f'/test{suffix}.txt'
        self.path = path
        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        allPos = []
        self.traindataSize = 0
        self.testDataSize = 0

        with open(train_file) as f:
            for l in tqdm(f.readlines()):
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items))
                    trainItem.extend(items)
                    allPos.append(np.array(items))
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.traindataSize += len(items)
                if uid==100:
                    if config['test']:
                        break
        
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)
        #with open(f'./data/cf/allPosItem{suffix}.pkl', 'rb') as f:
        #    self.allPosItem = pickle.load(f)
        with open(f'./data/cf/allPos{suffix}.pkl', 'rb') as f:
            self._allPos = pickle.load(f)            

        with open(test_file) as f:
            for l in tqdm(f.readlines()):
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    testUniqueUsers.append(uid)
                    testUser.extend([uid] * len(items))
                    testItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.testDataSize += len(items)
                if uid==100:
                    if config['test']:
                        break
        self.m_item += 1
        self.n_user += 1
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)
        
        self.Graph = None
        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        #print(f"{world.dataset} Sparsity : {(self.trainDataSize + self.testDataSize) / self.n_users / self.m_items}")

        # (users,items), bipartite graph
        #if config['model']=='lgn':
        #    self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
        #                                shape=(self.n_user, self.m_item))
            #self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
            #self.users_D[self.users_D == 0.] = 1
            #self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
            #self.items_D[self.items_D == 0.] = 1.
        # pre-calculate
        #self._allPos = allPos
        self.__testDict = self.__build_test()
        #print(f"{world.dataset} is ready to go")

    @property
    def n_users(self):
        return self.n_user
    
    @property
    def m_items(self):
        return self.m_item
    
    @property
    def trainDataSize(self):
        return self.traindataSize
    
    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def _split_A_hat(self,A):
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold*fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(self.device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
        
    def getSparseGraph(self):
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except :
                print("generating adjacency matrix")
                s = time()
                print(1)
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                print(2)
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()
                print(3)
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
                
                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
                print(4)
                
                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                print(5)
                end = time()
                print(f"costing {end-s}s, saved norm_mat...")
                sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)
                print(6)

            if self.split == True:
                self.Graph = self._split_A_hat(norm_adj)
                print("done split matrix")
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().to(self.device)
                print("don't split the matrix")
        return self.Graph

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.allPos[user])
            #posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems
    

class Datas:
    def __init__(self, config):
        suffix = config['suffix']
        product = pd.read_pickle(f'./data/cb/product_cb{suffix}.pkl')
        customer = pd.read_pickle(f'./data/cb/customer_cb{suffix}.pkl')
        train = pd.read_pickle(f'./data/train{suffix}.pkl')
        test = pd.read_pickle(f'./data/test{suffix}.pkl')
        if suffix=='all':
            inference = pd.read_pickle(f'./data/inference{suffix}.pkl')
        print('loaded data')
        self.customer = customer
        self.product = product
        self.n_user = len(customer)
        self.m_item = len(product)
        self.n_users = len(customer)
        self.m_items = len(product)
        print(self.n_user, self.m_items)
        self.trainUser = train['cf_customer'].values.astype(int)
        self.trainItem = train['cf_product'].values.astype(int)
        if suffix=='all':
            self.inferenceUser = inference['cf_customer'].values.astype(int)
            self.inferenceItem = inference['cf_product'].values.astype(int)
        else:
            self.inferenceUser = self.trainUser
            self.inferenceItem = self.trainItem
            
        self.testUser = test['cf_customer'].values.astype(int)
        self.testItem = test['cf_product'].values.astype(int)
        self.trainDataSize = len(train)
        with open(f'./data/cf/allPos{suffix}.pkl', 'rb') as f:
            self.allPos = pickle.load(f)
        self.testDict = self.__build_test()
        self.user_oc = torch.zeros(self.n_user).long()
        self.item_oc = torch.zeros(self.m_item).long()
        user_id, user_count=torch.unique(torch.from_numpy(self.trainUser).long(), return_counts=True)
        item_id, item_count=torch.unique(torch.from_numpy(self.trainItem).long(), return_counts=True)
        self.user_oc[user_id] = user_count
        self.item_oc[item_id] = item_count
            
    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.allPos[user])
            #posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems
    
    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {u: [] for u in range(self.n_users)}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data
        
            
        
def compute_offsets(neighbor_lens):
    cumsum = list(itertools.accumulate(neighbor_lens, operator.add))
    return [0] + cumsum[:-1]

def get_indice_offset(l):
    indice, lens = [], []
    for _ in l:
        indice.extend(_)
        lens.append(len(_))
    return indice, compute_offsets(lens)

def save_result(config, dataset, product_names,  users_list, rating_list, groundTrue_list):
    allPos = dataset.allPos
    train_names, test_names, predict_names = [], [], []
    train_ids, test_ids, predict_ids = [], [], []
    for i in users_list[0]:
        if len(allPos[i]):
            train_names.append(product_names[allPos[i].astype(int)].tolist())
        else:
            train_names.append([])
        if len(groundTrue_list[0][i]):
            test_names.append(product_names[np.array(groundTrue_list[0][i]).astype(int)].tolist())
        else:
            test_names.append([])
        if len(rating_list[0][i]):
            predict_names.append(product_names[np.array(rating_list[0][i]).astype(int)].tolist())
        else:
            product_names.append([])
        train_ids.append(allPos[i])
        test_ids.append(list(groundTrue_list[0][i]))
        predict_ids.append(list(rating_list[0][i].numpy()))
    
    customer_ids = users_list[0]
    train_names, test_names, predict_names = join_list(train_names), join_list(test_names), join_list(predict_names)
    train_ids, test_ids, predict_ids = join_list(train_ids), join_list(test_ids), join_list(predict_ids)
    dataframe = pd.DataFrame({'customer_id': customer_ids,
                                'train_name': train_names,
                                'predict_name': predict_names,
                                'gt_name': test_names,
                                'gt_id': test_ids,
                                'predict_id': predict_ids,
                                'train_id': train_ids})
    model, recdim, layer= config['model'], config['recdim'], config['layer']
    wandb = config['wandb']
    suffix = config['suffix']
    df_save_name = f'./data/result/{model}_{recdim}_{layer}_multi_process.csv'
    dataframe.to_csv(df_save_name)
    print('saved result')
        
class TextSAGE(torch.nn.Module):
    def __init__(self, config, dataset):
        super().__init__()
        init_time = time.time()
        suffix = config["suffix"]
        self.dataset = dataset
        self.n_user = self.dataset.n_user
        self.m_item = self.dataset.m_item
        trainUser, trainItem = self.dataset.trainUser, self.dataset.trainItem
        self.edge_index = torch.cat(
            [
                torch.stack(
                    [torch.tensor(trainUser).long(), torch.tensor(trainItem).long() + self.n_user],
                    dim=0,
                ),
                torch.stack(
                    [torch.tensor(trainItem).long() + self.n_user, torch.tensor(trainUser).long()],
                    dim=0,
                ),
            ],
            dim=1,
        )
        inferenceUser, inferenceItem = self.dataset.inferenceUser, self.dataset.inferenceItem
        self.inference_edge_index = torch.cat(
            [
                torch.stack(
                    [torch.tensor(inferenceUser).long(), torch.tensor(inferenceItem).long() + self.n_user],
                    dim=0,
                ),
                torch.stack(
                    [torch.tensor(inferenceItem).long() + self.n_user, torch.tensor(inferenceUser).long()],
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
        print('loaded category feature')
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
        print('loaded word embedding')
        print(time.time()-init_time)
        self.user_feature_num = self.user_features.shape[1]
        self.item_feature_num = self.item_features.shape[1]
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
        #self.item_sentence_embedding = torch.from_numpy(
        #    np.load(
        #        f"/home/yamanishi/project/furusato_recommend/data/cb/product_sentence_emb{suffix}.npy"
        #    )
        #).float()
        '''
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
        '''
        #with open(f'/home/yamanishi/project/furusato_recommend/data/customer_num_dask{suffix}.pkl', 'rb') as f:
        #    self.user_numeric_features = pickle.load(f)
        #with open(f'/home/yamanishi/project/furusato_recommend/data/product_num_dask{suffix}.pkl', 'rb') as f:
        #    self.item_numeric_features = pickle.load(f)
            
        self.user_numeric_features = torch.from_numpy(np.load(f'/home/yamanishi/project/furusato_recommend/data/cb/customer_numeric_numpy{suffix}.npy'))
        self.item_numeric_features = torch.from_numpy(np.load(f'/home/yamanishi/project/furusato_recommend/data/cb/product_numeric_numpy{suffix}.npy'))
        print('loaded numeric', time.time()-init_time)
        self.user_numeric_linear = torch.nn.Linear(
            self.user_numeric_features.shape[1], self.latent_dim
        )
        self.item_numeric_linear = torch.nn.Linear(
            self.item_numeric_features.shape[1], self.latent_dim
        )
        print(self.item_numeric_linear.weight.dtype)
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
        print("loaded text vecs", time.time()-init_time)
        self.vocab_num = self.item_name.shape[1]
        self.word_emb_dim = self.latent_dim // 2
        self.word_embedding = torch.nn.Embedding(
            num_embeddings=self.item_name.shape[1], embedding_dim=self.word_emb_dim
        )
        self.user_oc = dataset.user_oc
        self.item_oc = dataset.item_oc
        self.init_parameters()
        print('init parameters')

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
        text_embedding = [name_out, comment_out, comment_list_out]
        '''
        if mode=='item' and 'r' in self.config['item_feature']:
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
        '''
        text_embedding = torch.cat(text_embedding, dim=1)
        
        return text_embedding

    def get_initial_user_emb(self, user):
        # user: CPU
        #time1 = time.time()
        if 'n' in self.config['user_feature']:
            #numeric_embedding = self.user_numeric_linear(
            #    torch.from_numpy(self.user_numeric_features[user.numpy()].compute().todense()).float().to(self.device)
            #)
            numeric_embedding = self.user_numeric_linear(
                self.user_numeric_features[user].to(torch.float32).to(self.device)
            )
        #print(numeric_embedding)
        #print('numerical time', time.time()-time1)
        if 't' in self.config['user_feature']:
            text_embedding = self.get_text_embedding(user, mode="user")
        if 'w' in self.config['user_feature']:
            word_embedding = self.user_word_embedding[user].to(self.device)
        if 'c' in self.config['user_feature']:
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
            
        return user_embedding  # TODO: change

    def get_initial_item_emb(self, item):
        # item: CPU
        #time1 = time.time()
        if 'n' in self.config['item_feature']:
            #numeric_embedding = self.item_numeric_linear(
            #    torch.from_numpy(self.item_numeric_features[item.numpy()].compute().todense()).float().to(self.device)
            #)            
            numeric_embedding = self.item_numeric_linear(
                self.item_numeric_features[item].to(torch.float32).to(self.device)
            )
        #print('numerical time', time.time()-time1)
        if 't' in self.config['item_feature']:
            text_embedding = self.get_text_embedding(item, mode="item")
        #sentence_embedding = self.item_sentence_embedding[item].to(self.device)
        if 'w' in self.config['item_feature']:
            word_embedding = self.item_word_embedding[item].to(self.device)
        if 'c' in self.config['item_feature']:
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
            
        #if 's' in self.config['item_feature']:
        #    item_embedding.append(sentence_embedding)
                
        item_embedding = torch.cat(item_embedding, dim=1)
        
        item_embedding = self.item_proj(item_embedding)
        return item_embedding # TODO: change

    def get_initial_emb(self, index):
        user_index, item_index = (index < self.n_user),(index >= self.n_user)
        user, item = index[user_index], index[item_index]
        emb = torch.zeros((len(index), self.latent_dim * 1)).to(
            self.device
        )  # TODO: change
        # print(user, item)
        emb[user_index] = self.get_initial_user_emb(user)
        emb[item_index] = self.get_initial_item_emb(item - self.n_user)
        return emb

    def forward(self, x, adjs):
        # neighbors: [batch(item), batch*neighbor_size(user), batch*neighbor_size^2(item)]
        #          or [batch(user), batch*neighbor_size(item), batch*neighbor_size^2(user)]
        # offsets: [batch, batch*neighbor_size, batch*neighbor_size*neighbor_size]
        x_init = x[: adjs[-1][-1][1]]
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

    def OneEpoch(self, optimizer, user, pos, neg):
        print(len(user))
        pos = pos + self.n_user
        neg = neg + self.n_user
        # print(self.edge_index)
        # print(user, pos, neg)
        start_time = time.time()
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
        print('prepared loader', time.time()-start_time)
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
            optimizer.zero_grad()
            loss = self.loss(user_emb, pos_emb, neg_emb)
            aver_loss += loss.detach().cpu()
            loss.backward()
            # loss.backward(retain_graph=True)
            del loss, user_adjs, pos_adjs, neg_adjs, user_emb, pos_emb, neg_emb, user_id, pos_id, neg_id
            optimizer.step()
        aver_loss /= total_batch
        return aver_loss

    @torch.no_grad()
    def getUsersRating(self,):
        trainItem = torch.tensor(self.dataset.inferenceItem).to(self.device)
        trainUser = torch.tensor(self.dataset.inferenceUser).to(self.device)
        user_initial_emb, item_initial_emb = [], []
        for n in minibatch(torch.arange(self.n_user), batch_size=10000):
            user_initial_emb.append(self.get_initial_user_emb(n))
            
        for m in minibatch(torch.arange(self.m_item), batch_size=10000):
            item_initial_emb.append(self.get_initial_item_emb(m))
            
        user_initial_emb = torch.cat(user_initial_emb, dim=0)
        item_initial_emb = torch.cat(item_initial_emb, dim=0)

        user_x, item_x = user_initial_emb, item_initial_emb
        
        for i in range(self.num_layers):
            user_aggr, item_aggr = torch.zeros_like(user_x), torch.zeros_like(item_x)
            for trainUsertmp, trainItemtmp in minibatch(trainUser, trainItem, batch_size=500000):
                user_x_tmp, item_x_tmp = user_x[trainUsertmp], item_x[trainItemtmp]
                user_div = self.user_oc.to(self.device)[trainUsertmp]
                item_div = self.item_oc.to(self.device)[trainItemtmp]
                item_x_tmp = item_x_tmp/user_div.unsqueeze(1)
                user_x_tmp = user_x_tmp/item_div.unsqueeze(1)
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
            del user_aggr, item_aggr
        # print(user_x.shape, item_x.shape)
        #rating = torch.matmul(user_x[users], item_x.T)
        return user_x, item_x

        
def UniformSample( dataset, neg_ratio = 1):
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
        np.array
    """
    total_start = time.time()
    dataset : BasicDataset
    user_num = int(dataset.trainDataSize*TRAIN_ITERATIVE)
    #user_num = 100 #TODO: change
    sample_users = np.random.randint(0, len(dataset.allPos), user_num)
    allPos = dataset.allPos
    S = []
    sample_time1 = 0.
    sample_time2 = 0.
    neg_prob = dataset.item_oc.numpy()**NEGATIVE_POW
    neg_prob = neg_prob/sum(neg_prob)
    print(sum(neg_prob))
    oc_count = defaultdict(int)
    users, posItems, negItems = [],[],[]
    for i, user in tqdm(enumerate(sample_users)):
        start = time.time()
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        sample_time2 += time.time() - start
        posindex = np.random.randint(0, len(posForUser),)
        positem = posForUser[posindex]
        if oc_count[positem]>=POSITIVE_NUM_LIMIT:continue
        oc_count[positem]+=1
        while True:
            #negitem = np.random.choice(dataset.m_items, p=neg_prob)
            negitem = np.random.randint(0, dataset.m_items,)
            if negitem in posForUser:
                continue
            else:
                break
        S.append([user, positem, negitem])
        end = time.time()
        sample_time1 += end - start
    total = time.time() - total_start
    return np.array(S)    

def minibatch(*tensors, **kwargs):

    batch_size = kwargs.get('batch_size', 2048)

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)   
def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')
   
def test_one_batch(X):
    sorted_items = X[0].cpu().numpy()
    groundTrue = X[1]
    r = getLabel(groundTrue, sorted_items)
    pre, recall, ndcg, hr, diversity= [], [], [], [], []
    #print(world.topks)
    for k in [10, 20]:
        ret = RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        hr.append(ret['hr'])
        ndcg.append(NDCGatK_r(groundTrue,r,k))
        diversity.append(0.99)
        #diversity.append(Diversity(groundTrue, sorted_items, k, world.config['suffix']))
    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg),
            'hr': np.array(hr),
            'diversity': np.array(diversity)}
     
def demo(rank, world_size, gpu_index):
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    gpu = gpu_index[rank]
    rank = gpu
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)
    torch.cuda.manual_seed_all(42)
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(1000*rank)
    # create model and move it to GPU with id rank
    os.environ["OMP_NUM_THREADS"] = "4"
    config = {}
    config['device'] = f'cuda:{rank}'
    config['A_split'] = 1
    config['A_n_fold'] = 1
    config['model'] = 'textsage'
    config['wandb'] =False
    config['suffix'] = 'all'
    config['decay'] = 1e-6
    config['bpr_batch_size'] = 5000
    config['recdim']=32
    config['layer'] = 2
    config['num_neighbors']=5
    config['test_u_batch_size'] = 1000
    config['user_feature'] = 'nwt'
    config['item_feature'] = 'nwt'
    config['lr'] = 1e-3
    if config['wandb'] and rank==0:
        wandb.init('furusato_recommendation', name='multi process')
    dist.barrier()
    dataset = Datas(config)
    model = TextSAGE(config, dataset).to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    optimizer = optim.Adam(ddp_model.parameters(), lr=1e-3)
    S = UniformSample(dataset)
    product_names = dataset.product['name'].values
    user, pos, neg = torch.from_numpy(S[:, 0]), torch.from_numpy(S[:, 1]), torch.from_numpy(S[:, 2])
    #print(user, pos, neg)
    max_recall=0
    for i in range(200):
        dist.barrier()
        #TODO: change
        loss=ddp_model.module.OneEpoch(optimizer, user, pos, neg)
        print(loss)
        if rank==0 and config['wandb']:
            wandb.log({'loss': loss})
        if i%TEST_SPAN==0 and rank==0:
            ddp_model.eval()
            users = torch.arange(dataset.n_users)
            with torch.no_grad():
                user_x, item_x = ddp_model.module.getUsersRating()
                users_list, rating_list, groundTrue_list = [], [], []
                count=0
                for batch_users in tqdm(minibatch(users, batch_size=config['test_u_batch_size'])):
                    allPos = dataset.getUserPosItems(batch_users)
                    groundTrue = [dataset.testDict[u] for u in batch_users.numpy()]
                    batch_users_gpu = torch.Tensor(batch_users).long()
                    batch_users_gpu = batch_users_gpu.to(rank)
                    #rating = rating.cpu()
                    exclude_index = []
                    exclude_items = []
                    for range_i, items in enumerate(allPos):
                        exclude_index.extend([range_i] * len(items))
                        exclude_items.extend(items)
                    rating = torch.matmul(user_x[batch_users_gpu], item_x.T)
                    rating[exclude_index, exclude_items] = -(1<<10)
                    _, rating_K = torch.topk(rating, k=max([10, 20]))
                    rating = rating.cpu().numpy()
                    users_list.append(batch_users)
                    rating_list.append(rating_K.cpu())
                    groundTrue_list.append(groundTrue)
                    count+=1
                    del rating, batch_users_gpu, rating_K
                    if count==TEST_COUNT:
                        break
                del user_x, item_x
                X = zip(rating_list, groundTrue_list)
                #if multiprocessing.get_start_method() == 'fork':
                #    multiprocessing.set_start_method('spawn', force=True)
                #    print("{} setup done".format(multiprocessing.get_start_method()))
                pre_results = []
                for x in X:
                    pre_results.append(test_one_batch(x))
                topks=[10, 20]
                metrics = ['precision', 'recall', 'ndcg', 'hr', 'coverage', 'diversity']
                results = {metric: np.zeros(len(topks)) for metric in metrics}
                for result in pre_results:
                    for metric in result.keys():
                        results[metric]+=result[metric]
                for metric in results.keys():
                    results[metric]/=dataset.n_user
                for i,k in enumerate(topks):
                    #novelty = Novelty(rating_list, dataset.n_users, k, config['suffix'], ddp_model.module.item_oc)
                    coverage = Coverage(rating_list, dataset.m_items, k)
                    #results['novelty'][i] = novelty/dataset.n_user
                    results['coverage'][i] = coverage
                if results['recall'][0]>max_recall:
                    max_recall = results['recall'][0]
                print(results)    
                metrics = results
                for i, k in enumerate([10, 20]):
                    metrics_tmp = {f'{metric}@{k}': metrics[metric][i] for metric in metrics.keys()}
                    if config['wandb']:
                        wandb.log(metrics_tmp)
                        
                save_result(config, dataset, product_names, users_list, rating_list, groundTrue_list)
            del users

    cleanup()
    
def setup(rank, world_size):
    #os.environ['MASTER_ADDR'] = '172.16.2.15'
    #os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '23462'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
def cleanup():
    dist.destroy_process_group()
    
if __name__=='__main__':
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    print(world_size)
    world_size=2
    gpu_index = [i for i in range(world_size)]
    mp.spawn(demo,
             args=(world_size, gpu_index),
             nprocs=world_size,
             join=True)