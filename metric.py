import torch
from typing import Dict, List, Union
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score
from dataloader import BasicDataset
import itertools
from sklearn.utils.extmath import cartesian

class Metric:
    def __init__(self, config, dataset):
        self.config = config
        #self.dataset = dataset
        self.device = config['device']
        
    def calc_metrics(self, rating: torch.Tensor, test_item: Dict[int,int], k=10) -> Dict[str,float]:
        _, rating_K = torch.topk(rating, k=k)
        #calculate recall, precision, diversity, novelty
        recall, prec = 0, 0
        for i, rk in enumerate(rating):
            gt = test_item[i]
            r = torch.isin(torch.tensor(gt).to(self.device), rk).sum()
            
            prec += r/k
            recall += r/len(gt)
        
        prec/=len(rating_K)
        recall/=len(rating_K)
        
        return {'recall': recall,
                'precision': prec,
                'length': len(rating)}
        
    @staticmethod
    def aggregate_metrics(metrics_all: List[Dict[str,float]])->Dict[str,float]:
        aggregate_dict = defaultdict(float)
        total_length = 0
        for metrics in metrics_all:
            total_length+=metrics['length']
            for k,v in metrics.items():
                if k!='length':
                    aggregate_dict[k]+=v*metrics['length']
                    
        aggregate_dict = {k:v/total_length for k,v in aggregate_dict.items()}
        return aggregate_dict           
            
        
def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred/recall_n)
    precis = np.sum(right_pred)/precis_n
    hr = np.sum(right_pred>=1)
    return {'recall': recall, 'precision': precis, 'hr': hr}

def MRRatK_r(r, k):
    """
    Mean Reciprocal Rank
    """
    pred_data = r[:, :k]
    scores = np.log2(1./np.arange(1, k+1))
    pred_data = pred_data/scores
    pred_data = pred_data.sum(1)
    return np.sum(pred_data)

def NDCGatK_r(test_data,r,k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1./np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data*(1./np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg/idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)

def Diversity(groundTrue, sorted_items, k, suffix):
    #product_categories = np.load(f'./data/cb/product_categories{suffix}.npy', allow_pickle=True)
    product_categories = np.load('./data/cb/product_categories.npy', allow_pickle=True)
    diversity = 0
    for items in sorted_items:
        for i in range(k):
            for j in range(i+1, k):
                dist = (1-len(set(product_categories[items[i]])&set(product_categories[items[j]]))/(len(set(product_categories[items[i]])|set(product_categories[items[j]]))+1e-6))
                diversity+=dist
    diversity/=((k-1)*k//2)
    return diversity

def Novelty(sorted_items, n_users, k, suffix):
    #oc = np.load(f'./data/cf/product_occurance{suffix}.npy')/n_users
    oc = np.load('./data/cf/product_occurance.npy')/n_users
    total_novelty=0
    for batch_items in sorted_items:
        for items in batch_items:
            total_novelty+=np.sum(-np.log2(oc[items[:k]]))/k
    return total_novelty/-np.log2(1/n_users)

def Unexpectedness(allPos, rating_list, pmi, k):
    return 1
    unexp=0
    i=0
    for rating_batch in rating_list:
        for rating in rating_batch:
            gt = allPos[i]
            prod = cartesian(gt, rating)
            for p in prod:
                unexp+=pmi[p]
            unexp/=len(prod)
            i+=1
            
    return unexp

def Coverage(rating_list, m_items, k):
    item_set = set()
    for rating_batch in rating_list:
        for rating in rating_batch:
            item_set.update(set(rating[:k].tolist()))
    return len(item_set)/m_items

def AUC(all_item_scores, dataset, test_data):
    """
        design for a single user
    """
    dataset : BasicDataset
    r_all = np.zeros((dataset.m_items, ))
    r_all[test_data] = 1
    r = r_all[all_item_scores >= 0]
    test_item_scores = all_item_scores[all_item_scores >= 0]
    return roc_auc_score(r, test_item_scores)

