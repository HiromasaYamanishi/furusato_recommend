from dataloader import BasicDataset, Loader
import numpy as np
from time import time
from tqdm import tqdm
from multiprocessing import Pool, Process, Manager
import torch.multiprocessing as multiprocessing

class UniformSampling:
    def __init__(self, dataset: Loader, neg_ratio=1):
        #self.dataset = dataset
        self.process_num = 8
        self.m_items = dataset.m_items
        self.n_users = dataset.n_user
        self.user_num = dataset.trainDataSize
        self.allPos = dataset.allPos
        
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
            posindex = np.random.randint(0, len(posForUser))
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
    process_num=8
    start_index = (sample_users//process_num)*i
    end_index = min(sample_users//process_num)*(i+1)
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
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        while True:
            negitem = np.random.randint(0, dataset.m_items)
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
    total = time() - total_start
    return np.array(users), np.array(posItems), np.array(negItems)
