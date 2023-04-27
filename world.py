import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
import multiprocessing
from enum import Enum
from os.path import join

import torch

from parse import parse_args

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
args = parse_args()

ROOT_PATH = "/home/yamanishi/project/furusato_recommend"
CODE_PATH = join(ROOT_PATH, 'code')
DATA_PATH = join(ROOT_PATH, 'data')
BOARD_PATH = join(CODE_PATH, 'runs')
FILE_PATH = join(CODE_PATH, 'checkpoints')
import sys

sys.path.append(join(CODE_PATH, 'sources'))


#if not os.path.exists(FILE_PATH):
#    os.makedirs(FILE_PATH, exist_ok=True)


config = {}
all_dataset = ['furusato']
all_models  = ['mf', 'lgn', 'sage', 'radj', 'pinsage', 'textsage', 'textsage_id', 
               'tgrec', 'tgsrec', 'lightsage', 'tgrec2', 'pinsage', 'fsage', 'rsage','sasgnn']
# config['batch_size'] = 4096
config['bpr_batch_size'] = args.bpr_batch
config['latent_dim_rec'] = args.recdim
config['lightGCN_n_layers']= args.layer
config['dropout'] = args.dropout
config['keep_prob']  = args.keepprob
config['A_n_fold'] = args.a_fold
config['test_u_batch_size'] = args.testbatch
config['multicore'] = args.multicore
config['lr'] = args.lr
config['decay'] = args.decay
config['pretrain'] = args.pretrain
config['A_split'] = True
config['bigdata'] = False
device = args.device
config['device'] = f'cuda:{device}'
#config['device'] = 'cpu'
config['multi_gpu'] = args.multi_gpu
config['test'] = args.test
config['model'] = args.model
config['recdim'] = args.recdim
config['layer'] = args.layer
config['num_neighbors'] = args.num_neighbors
config['wandb'] = args.wandb
config['inference'] = args.inference
config['train_emb'] = args.train_emb
config['sample_pow'] = args.sample_pow
config['test_span'] = args.test_span
config['r'] = args.r
config['multicore'] = args.multicore
config['suffix'] = args.suffix
config['multi_relational'] = args.multi_relational
config['conv'] = args.conv
config['for_lgbm'] = args.for_lgbm
config['lgbm_ratio'] = args.lgbm_ratio
config['cold_start'] = args.cold_start
config['user_feature'] = args.user_feature
config['item_feature'] = args.item_feature
config['factorization'] = args.factorization

if config['model'] == 'textsage':
    assert len(args.user_feature)>0
    assert len(args.item_feature)>0
    '''
    feature set
    n: numeric feature
    c: category feature
    w: word embedding feature
    t: text feature
    s: sentence feature
    r: review feature
    b: bert feature
    '''
    for f in args.user_feature:
        if f not in 'ncwtbs':
            raise ValueError
        
    for f in args.item_feature:
        if f not in 'ncwtsrb':
            raise ValueError
            
#GPU = torch.cuda.is_available()
#device = torch.device('cuda' if GPU else "cpu")
CORES = multiprocessing.cpu_count() // 2
seed = args.seed

dataset = args.dataset
model_name = args.model
if dataset not in all_dataset:
    raise NotImplementedError(f"Haven't supported {dataset} yet!, try {all_dataset}")
#if model_name not in all_models:
#    raise NotImplementedError(f"Haven't supported {model_name} yet!, try {all_models}")




TRAIN_epochs = args.epochs
LOAD = args.load
PATH = args.path
topks = eval(args.topks)
tensorboard = args.tensorboard
comment = args.comment
# let pandas shut up
from warnings import simplefilter

simplefilter(action="ignore", category=FutureWarning)



def cprint(words : str):
    print(f"\033[0;30;43m{words}\033[0m")

logo = r"""
██╗      ██████╗ ███╗   ██╗
██║     ██╔════╝ ████╗  ██║
██║     ██║  ███╗██╔██╗ ██║
██║     ██║   ██║██║╚██╗██║
███████╗╚██████╔╝██║ ╚████║
╚══════╝ ╚═════╝ ╚═╝  ╚═══╝
"""