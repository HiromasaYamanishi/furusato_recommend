import itertools
import operator
import numpy as np
from multiprocessing import Manager
import torch
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def compute_offsets(neighbor_lens):
    cumsum = list(itertools.accumulate(neighbor_lens, operator.add))
    return [0] + cumsum[:-1]

def uniform_neighbors(nodes, dataset, num_neighbors, mode):
    if mode=='user':
        allPos = dataset.allPos
    else:
        allPos = dataset.allPosItem

    neighbors, offsets = [], []
    for n in nodes:
        #print(n, mode, len(allPos))
        if len(allPos[n])>0:
            samples = np.random.choice(allPos[n], num_neighbors, replace=True)
        else:
            samples = np.random.choice(np.arange(len(allPos)), num_neighbors, replace=True)
        neighbors.extend(samples.tolist())
        offsets.append(len(neighbors))
        
    return neighbors, [0]+offsets[:-1]


class UniformNeighbors:
    def __init__(self, dataset, num_layers, num_neighbors):
        self.dataset = dataset
        self.num_layers = num_layers
        self.num_neighbors = num_neighbors
        self.process_num = 16
    
    def uniform_neighbors(self, nodes, dataset, num_neighbors, mode):
        if mode=='user':
            allPos = dataset.allPos
        else:
            allPos = dataset.allPosItem

        neighbors, offsets = [], []
        for n in nodes:
            #print(n, mode, len(allPos))
            if len(allPos[n])>0:
                samples = np.random.choice(allPos[n], num_neighbors, replace=True)
            else:
                samples = np.random.choice(np.arange(len(allPos)), num_neighbors, replace=True)
            neighbors.extend(samples.tolist())
            offsets.append(len(neighbors))
            
        return neighbors, [0]+offsets[:-1]
    
    def sample_neighbors(self, nodes, mode):
        neighbors, offsets = [], []
        neighbors.append(nodes)
        offsets.append(list(torch.arange(len(nodes))))
        for i in range(self.num_layers):
            neighbor, offset = self.uniform_neighbors(nodes, self.dataset, self.num_neighbors, mode)
            neighbors.append(neighbor)
            offsets.append(offset)
            nodes = neighbor
            if mode=='user':
                mode='item'
            else:mode='user'
        return neighbors, offsets
    
    def sample_parallel(self, user, pos, neg):
        user_neighbors, user_offsets = self.sample_neighbors(user, mode='user')
        pos_neighbors, pos_offsets = self.sample_neighbors(pos, mode='item')
        neg_neighbors, neg_offsets = self.sample_neighbors(neg, mode='item')
        #return_dict[process_index] = (user_neighbors, user_offsets, pos_neighbors, pos_offsets, neg_neighbors, neg_offsets)
        return (user_neighbors, user_offsets, pos_neighbors, pos_offsets, neg_neighbors, neg_offsets)
        #print('sample parallel done')
        
    def sample(self, user, pos, neg):
        process_unit=len(user)//self.process_num
        user_all, pos_all, neg_all = [], [], []
        for i in range(self.process_num):
            user_all.append(user[process_unit*i:process_unit*(i+1)])
            pos_all.append(pos[process_unit*i:process_unit*(i+1)])
            neg_all.append(neg[process_unit*i:process_unit*(i+1)])
        future_list = []
        with ProcessPoolExecutor(max_workers=self.process_num) as executor:
            for i in range(self.process_num):
                future = executor.submit(self.sample_parallel, user=user[process_unit*i:process_unit*(i+1)], pos=pos[process_unit*i:process_unit*(i+1)], neg=neg[process_unit*i:process_unit*(i+1)])
                future_list.append(future)
        result = [f.result() for f in future_list]
        #print(result)
        #result = [f.result() for f in futures]
        return result
        
    def sample_(self, user, pos, neg):
        torch.multiprocessing.set_start_method('fork', force=True)
        manager = Manager()
        return_dict = manager.dict()
        process_list = []
        process_unit = len(user)//self.process_num
        for i in range(self.process_num):
            #print(i)
            process = torch.multiprocessing.Process(
                target = self.sample_parallel,
                kwargs = {'process_index': i,
                          'user': user[process_unit*i:process_unit*(i+1)],
                          'pos': pos[process_unit*i:process_unit*(i+1)],
                          'neg': neg[process_unit*i:process_unit*(i+1)],
                          'return_dict': return_dict}
            )
            process.start()
            process_list.append(process)
            
        for process in process_list:
            process.join()
            
        return return_dict
        
    
                
                