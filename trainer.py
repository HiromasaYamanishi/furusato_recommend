from tqdm import tqdm
import time
import copy
import math
import os
import yaml
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import minibatch, shuffle, getLabel
from negative_sample import UniformSample, UniformSampling
import world
from metric import Metric,RecallPrecision_ATk, NDCGatK_r
from multiprocessing import Manager, Pool, Process
import torch.multiprocessing as multiprocessing
import numpy as np
import utils
import wandb


class Trainer:
    def __init__(self, config, dataset, model):
        self.config = config
        self.dataset = dataset
        self.metric = Metric(config, dataset)
        self.sampling = UniformSampling(dataset)
        self.device = self.config['device']
        self.model = model
        self.process_num = 8
        self.max_recall = 0
        #self.model.to(self.device)
        if world.config['multi_gpu']:
            self.model = torch.nn.DataParallel(self.model, device_ids=[1,2,3])
        else:
            self.model = self.model.to(self.device)
        print(model)
        wandb.init('furusato_recommendation', name=config['wandb_name'])
        
        
    def train(self):
        S = self.sampling.sample()
        users, posItems, negItems = torch.tensor(S[:, 0]).to(self.device), torch.tensor(S[:, 1]).to(self.device), torch.tensor(S[:, 2]).to(self.device)
        #users, posItems, negItems = UniformSample(self.dataset)
        users, posItems, negItems = shuffle(users, posItems, negItems)
        aver_loss = 0
        total_batch = len(users)//world.config['bpr_batch_size'] + 1
        for batch_i, (user, posItem, negItem) in tqdm(enumerate(minibatch(users, posItems, negItems, batch_size=world.config['bpr_batch_size']))):
            if world.config['multi_gpu']:
                loss = self.model.module.stageOne(user, posItem, negItem)
            else:
                user, posItem, negItem = user.to(self.device), posItem.to(self.device), negItem.to(self.device)
                loss = self.model.stageOne(user, posItem, negItem)
            aver_loss+=loss
        aver_loss/=total_batch
        
        return aver_loss
    
    def test(self):
        self.model.eval()
        dataset = self.dataset
        users = list(dataset.testDict.keys())
        u_batch_size = world.config['test_u_batch_size']
        users_list, rating_list, groundTrue_list = [], [], []
        total_batch = len(users) // u_batch_size + 1
        with torch.no_grad():
            for batch_users in tqdm(minibatch(users, batch_size=u_batch_size)):
                allPos = dataset.getUserPosItems(batch_users)
                groundTrue = [dataset.testDict[u] for u in batch_users]
                batch_users_gpu = torch.Tensor(batch_users).long()
                batch_users_gpu = batch_users_gpu.to(world.device)

                rating = self.model.getUsersRating(batch_users_gpu)
                #rating = rating.cpu()
                exclude_index = []
                exclude_items = []
                for range_i, items in enumerate(allPos):
                    exclude_index.extend([range_i] * len(items))
                    exclude_items.extend(items)
                rating[exclude_index, exclude_items] = -(1<<10)
                _, rating_K = torch.topk(rating, k=20)
                rating = rating.cpu().numpy()
                del rating
                users_list.append(batch_users)
                rating_list.append(rating_K.cpu())
                groundTrue_list.append(groundTrue)
            assert total_batch == len(users_list)
            X = zip(rating_list, groundTrue_list)
            if multiprocessing.get_start_method() == 'fork':
                multiprocessing.set_start_method('spawn', force=True)
                print("{} setup done".format(multiprocessing.get_start_method()))
            start_time = time.time()
            
            pool = multiprocessing.Pool(self.process_num)
            pre_results = pool.map(test_one_batch, X)
            results = {'precision': np.zeros(len(world.topks)),
                    'recall': np.zeros(len(world.topks)),
                    'ndcg': np.zeros(len(world.topks)),
                    'hr': np.zeros(len(world.topks))}
            scale = float(u_batch_size/len(users))
            print(time.time()- start_time)
            for result in pre_results:
                results['recall'] += result['recall']
                results['precision'] += result['precision']
                results['ndcg'] += result['ndcg']
                results['hr'] += result['hr']
            results['recall'] /= float(len(users))
            results['precision'] /= float(len(users))
            results['ndcg'] /= float(len(users))
            results['hr'] /= float(len(users))
            
            if results['recall'][0]>self.max_recall:
                self.max_recall = results['recall'][0]
                self.save_model()
                
            
            return results
        
    def save_model(self):
        model, recdim, layer = self.config['model'], self.config['recdim'], self.config['layer']
        save_dir=os.path.join('/home/yamanishi/project/furusato_recommend/checkpoints', model)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f'{recdim}_{layer}.pth')
        torch.save(self.model.state_dict(), save_path)
        print('saved model')
        
    
    def train_epoch(self):
        for epoch in range(world.TRAIN_epochs):
            loss = self.train()
            print(loss)
            wandb.log({'loss':loss.item()})
            if epoch%5==0:
                metrics = self.test()
                print(metrics)
                wandb.log({'hr@20': metrics['hr'][0],
                           'recall@20': metrics['recall'][0],
                           'precision@20': metrics['precision'][0],
                           'ndcg': metrics['ndcg'][0]})
        
                
            
def test_one_batch(X):
    sorted_items = X[0].cpu().numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg, hr = [], [], [], []
    for k in world.topks:
        ret = RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        hr.append(ret['hr'])
        ndcg.append(NDCGatK_r(groundTrue,r,k))
    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg),
            'hr': np.array(hr)}