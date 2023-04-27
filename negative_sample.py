import math
import pickle
from multiprocessing import Manager, Pool, Process
from time import time

import numpy as np
import torch.multiprocessing as multiprocessing
from tqdm import tqdm

from dataloader import BasicDataset, Loader


class UniformSampling:
    def __init__(self, dataset: Loader, config, neg_ratio=1):
        #self.dataset = dataset
        self.process_num = 4
        self.m_items = dataset.m_items
        self.n_users = dataset.n_user
        self.user_num = dataset.trainDataSize
        self.allPos = dataset.allPos
        self.config = config
        if config['sample_pow']==0.1:
            with open('./data/sample_prob/sample_prob_01.pkl', 'rb') as f:
                self.probs = pickle.load(f)
                
        elif config['sample_pow'] ==0.2:
            with open('./data/sample_prob/sample_prob_02.pkl', 'rb') as f:
                self.probs = pickle.load(f)
                
        elif config['sample_pow'] ==0.5:
            with open('./data/sample_prob/sample_prob_05.pkl', 'rb') as f:
                self.probs = pickle.load(f)
                
        elif config['sample_pow'] ==1:
            with open('./data/sample_prob/sample_prob_10.pkl', 'rb') as f:
                self.probs = pickle.load(f)
                
                
        
    def sample_parallel(self, process_index, process_num, sample_users, allPos, returned_dict):
        start_index = (len(sample_users)//process_num)*process_index
        end_index = (len(sample_users)//process_num)*(process_index+1)
        S = []
        sample_time1 = 0.
        sample_time2 = 0.
        users, posItems, negItems = [],[],[]
        for i, user in tqdm(enumerate(sample_users[start_index:end_index])):
            start = time()
            posForUser = allPos[user]
            if len(posForUser) == 0:
                continue
            sample_time2 += time() - start
            if self.config['sample_pow']==0:
                posindex = np.random.randint(0, len(posForUser))
            else:
                posindex = np.random.choice(len(posForUser), p=self.probs[user])
            positem = posForUser[posindex]
            while True:
                negitem = np.random.randint(0, self.m_items)
                if negitem in posForUser:
                    continue
                else:
                    break
            users.append(user)
            posItems.append(positem)
            negItems.append(negitem)
            S.append([user, positem, negitem])
            end = time()
            sample_time1 += end - start
        returned_dict[process_index] = np.array(S)
        #return np.array(S)
        
    
    def sample(self):
        user_num = self.user_num
        sample_users = np.random.randint(0, self.n_users, user_num)
        allPos = self.allPos
        manager = Manager()
        returned_dict = manager.dict()
        process_list = []
        for i in range(self.process_num):
            process = multiprocessing.Process(
                target = self.sample_parallel,
                kwargs={'process_index':i,
                        'process_num':self.process_num,
                        'sample_users': sample_users,
                        'allPos': allPos,
                        'returned_dict': returned_dict}
            )
            process.start()
            process_list.append(process)
            
        for process in process_list:
            process.join()
            
        return np.concatenate(returned_dict.values())

def UniformSample(dataset: Loader, neg_ratio = 1):
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
        np.array
    """
    total_start = time()
    dataset : BasicDataset
    user_num = dataset.trainDataSize
    sample_users = np.random.randint(0, dataset.n_users, user_num)
    allPos = dataset.allPos
    S = []
    sample_time1 = 0.
    sample_time2 = 0.
    users, posItems, negItems = [],[],[]
    for i, user in tqdm(enumerate(sample_users)):
        start = time()
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        sample_time2 += time() - start
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        while True:
            negitem = np.random.randint(0, dataset.m_items)
            if negitem in posForUser:
                continue
            else:
                break
        #users.append(user)
        #posItems.append(positem)
        #negItems.append(negitem)
        S.append([user, positem, negitem])
        end = time()
        sample_time1 += end - start
    total = time() - total_start
    return np.array(S)
