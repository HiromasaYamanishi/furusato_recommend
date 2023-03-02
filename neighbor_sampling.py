import itertools
import operator
import numpy as np

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
        
    
                
                