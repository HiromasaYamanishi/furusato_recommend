## for docker
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
from metric import Metric,RecallPrecision_ATk, NDCGatK_r, Diversity, Coverage, Novelty, Unexpectedness
from multiprocessing import Manager, Pool, Process
import torch.multiprocessing as multiprocessing
import numpy as np
import pandas as pd
import utils
from utils import join_list
import wandb


class Trainer:
    def __init__(self, config, dataset, model):
        self.config = config
        self.dataset = dataset
        self.metric = Metric(config, dataset)
        self.sampling = UniformSampling(dataset, config)
        print('setup sample')
        self.device = self.config['device']
        print(self.device)
        self.model = model
        self.process_num = 4
        self.max_recall = 0
        self.pmi = 0
        #self.model.to(self.device)
        if world.config['multi_gpu']:
            self.model = torch.nn.DataParallel(self.model, device_ids=[1,2,3])
        else:
            self.model = self.model.to(self.device)
        self.suffix = world.config['suffix']
        suffix = self.suffix
        self.product_names = np.load(f'./data/product_names{suffix}.npy', allow_pickle=True)
        self.categories = np.load(f'./data/cb/product_categories{suffix}.npy', allow_pickle=True)
        print(model)
        if not world.config['test']:
            wandb.init('furusato_recommendation', name=config['wandb'])
        #if multiprocessing.get_start_method() == 'fork':
        #    multiprocessing.set_start_method('spawn', force=True)
        
        
    def train(self):
        #if multiprocessing.get_start_method()=='spawn':
        #    multiprocessing.set_start_method('fork', force=True)
        if not self.config['multicore']:
            S = UniformSample(self.dataset)
        else:
            S = self.sampling.sample()
        users, posItems, negItems = torch.tensor(S[:, 0]).to(self.device), torch.tensor(S[:, 1]).to(self.device), torch.tensor(S[:, 2]).to(self.device)
        #users, posItems, negItems = UniformSample(self.dataset)
        users, posItems, negItems = shuffle(users, posItems, negItems)
        
        loss = self.model.OneEpoch(users, posItems, negItems)
        '''
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
        '''

        return loss
    
    def test(self):
        self.model.eval()
        dataset = self.dataset
        users = list(dataset.testDict.keys())
        u_batch_size = world.config['test_u_batch_size']
        users_list, rating_list, groundTrue_list = [], [], []
        total_batch = len(users) // u_batch_size + 1
        with torch.no_grad():
            if not self.config['inference']=='sample':
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
                    _, rating_K = torch.topk(rating, k=max(world.topks))
                    rating = rating.cpu().numpy()
                    del rating
                    users_list.append(batch_users)
                    rating_list.append(rating_K.cpu())
                    groundTrue_list.append(groundTrue)
                    #assert total_batch == len(users_list)
                    
            else:
                users_list, groundTrue_list, rating_list = self.model.getUsersRating(users)
            start_time = time.time()
            #print(rating_list, groundTrue_list)
            pre_results = []
            X = zip(rating_list, groundTrue_list)
            #if multiprocessing.get_start_method() == 'fork':
            #    multiprocessing.set_start_method('spawn', force=True)
            #    print("{} setup done".format(multiprocessing.get_start_method()))
            if not self.config['multicore']:
                pre_results = []
                for x in X:
                    pre_results.append(test_one_batch(x))
            else:
                pool = multiprocessing.Pool(self.process_num)
                pre_results = pool.map(test_one_batch, X)
            metrics = ['precision', 'recall', 'ndcg', 'hr', 'diversity', 'novelty', 'unexpectedness', 'coverage']
            results = {metric: np.zeros(len(world.topks)) for metric in metrics}
            scale = float(u_batch_size/len(users))
            print(time.time()- start_time)
            for result in pre_results:
                for metric in result.keys():
                    results[metric]+=result[metric]
            for metric in results.keys():
                results[metric]/=float(len(users))
            for i,k in enumerate(world.topks):
                novelty = Novelty(rating_list, self.dataset.n_users, k, self.suffix)
                coverage = Coverage(rating_list, self.dataset.m_items, k)
                unexpectedness = Unexpectedness(self.dataset.allPos, rating_list, self.pmi, k)
                results['novelty'][i] = novelty/float(len(users))
                results['coverage'][i] = coverage
                results['unexpectedness'][i] = unexpectedness/float(len(users))
            if results['recall'][0]>self.max_recall:
                self.max_recall = results['recall'][0]
                self.save_model()
            self.save_result(users_list, rating_list, groundTrue_list)
            self.model.test_item_emb = None
            print(time.time()- start_time)
            return results
        
    def save_result(self, users_list, rating_list, groundTrue_list):
        allPos = self.dataset.allPos
        train_names, test_names, predict_names = [], [], []
        train_ids, test_ids, predict_ids = [], [], []
        for i in users_list[0]:
            train_names.append(self.product_names[allPos[i]].tolist())
            test_names.append(self.product_names[np.array(groundTrue_list[0][i])].tolist())
            predict_names.append(self.product_names[np.array(rating_list[0][i])].tolist())
            train_ids.append(allPos[i].tolist())
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
        model, recdim, layer, inference = self.config['model'], self.config['recdim'], self.config['layer'], self.config['inference']
        suffix = self.config['suffix']
        sample_pow, r = self.config['sample_pow'], self.config['r']
        df_save_name = f'./data/result/{model}_{recdim}_{layer}_{sample_pow}_{r}_{suffix}.csv'
        dataframe.to_csv(df_save_name)
        print('saved result')
            
            
        
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
            if not self.config['test']:
                wandb.log({'loss':loss.item()})
            if epoch%self.config['test_span']==0:
                metrics = self.test()
                print(metrics)
                if not self.config['test']:
                    for i, k in enumerate(world.topks):
                        metrics_tmp = {f'{metric}@{k}': metrics[metric][i] for metric in metrics.keys()}
                        wandb.log(metrics_tmp)

        
                
            
def test_one_batch(X):
    sorted_items = X[0].cpu().numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg, hr, diversity= [], [], [], [], []
    #print(world.topks)
    for k in world.topks:
        ret = RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        hr.append(ret['hr'])
        ndcg.append(NDCGatK_r(groundTrue,r,k))
        diversity.append(Diversity(groundTrue, sorted_items, k, world.config['suffix']))
    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg),
            'hr': np.array(hr),
            'diversity': np.array(diversity)}