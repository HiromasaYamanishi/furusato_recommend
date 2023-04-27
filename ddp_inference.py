import itertools
import operator
import os
import pickle
import random
import gc
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
from utils import join_list, minibatch

import wandb
from metric import (Coverage, Diversity, Metric, NDCGatK_r, Novelty,
                    RecallPrecision_ATk, Unexpectedness, getLabel)

from ddp import Datas, TextSAGE
from logging import getLogger
logger = getLogger(__name__)

NEGATIVE_POW=0.2
POSITIVE_NUM_LIMIT=3000
TRAIN_ITERATIVE=3
TEST_COUNT=100
TEST_SPAN=5

        
            
        


def save_result(config, dataset, product_names,  users_list, rating_list, groundTrue_list, index=0):
    allPos = dataset.allPos
    train_names, test_names, predict_names = [], [], []
    train_ids, test_ids, predict_ids = [], [], []
    for i,user in enumerate(users_list[index]):
        user = user.item()
        if len(allPos[user]):
            train_names.append(product_names[allPos[user].astype(int)].tolist())
        else:
            train_names.append([])
        if len(groundTrue_list[index][i]):
            test_names.append(product_names[np.array(groundTrue_list[index][i]).astype(int)].tolist())
        else:
            test_names.append([])
        if len(rating_list[index][i]):
            predict_names.append(product_names[np.array(rating_list[index][i]).astype(int)].tolist())
        else:
            product_names.append([])
        train_ids.append(allPos[user])
        test_ids.append(list(groundTrue_list[index][i]))
        predict_ids.append(list(rating_list[index][i].numpy()))
    
    customer_ids = users_list[index]
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
    df_save_name = f'./data/result/{model}_{recdim}_{layer}_{index}_multi_process.csv'
    dataframe.to_csv(df_save_name)
    print('saved result')
        

   
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

    
if __name__=='__main__':
    DEVICE = 1
    USER_BATCH_SIZE = 1000

    torch.cuda.manual_seed_all(42)
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(1000*DEVICE)
    # create model and move it to GPU with id rank
    
    os.environ["OMP_NUM_THREADS"] = "4"
    config = {}
    config['device'] = f'cuda:{DEVICE}'
    config['A_split'] = 1
    config['A_n_fold'] = 1
    config['model'] = 'textsage'
    config['wandb'] =True
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

    dataset = Datas(config)
    logger.info('setup data')
    model = TextSAGE(config, dataset).to('cpu')
    suffix = config['suffix']
    model.load_state_dict(torch.load(f'/home/yamanishi/project/furusato_recommend/checkpoints/ddp_sage_{suffix}.pth'))
    model = model.to(config['device'])
    logger.info('setup model')

    user_x, item_x = model.getUsersRating()
    
    users = torch.arange(dataset.n_users)
    users_list, rating_list, groundTrue_list = [], [], []
    count=-1
    target_user_batch = [1000, 5000, 8500]
    for batch_users in tqdm(minibatch(users, batch_size=config['test_u_batch_size'])):
        count+=1
        if count not in target_user_batch:continue
        allPos = dataset.getUserPosItems(batch_users)
        groundTrue = [dataset.testDict[u] for u in batch_users.numpy()]
        batch_users_gpu = torch.Tensor(batch_users).long()
        batch_users_gpu = batch_users_gpu.to(config['device'])
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
        del rating, batch_users_gpu, rating_K
        #if count==TEST_COUNT:
        #    break
    del user_x, item_x
    print(len(groundTrue_list))
    print(len(rating_list))
    print(len(users_list))
    product_names = dataset.product['name'].values   
    for i in range(len(target_user_batch)):
        save_result(config, dataset, product_names, users_list, rating_list, groundTrue_list, i)