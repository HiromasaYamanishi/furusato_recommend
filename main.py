import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
import world
from dataloader import Loader
from model.graphsage import GraphSAGE
from model.lgcn import LightGCN
from model.lightsage import LightSAGE
from model.MF import MF  # , LightGCN
from model.pinsage import PinSAGE
from model.radj import rAdjGCN
from model.textsage import TextSAGE
from model.textsage_id import TextSAGE_ID
from model.tgrec import TGRec
from model.tgrec2 import TGRec2
from model.tgsrec import TGSRec
from model.fsage import FSAGE
from model.rsage import RSAGE
from model.sasgnn import SASGNN
from model.gnn import GNN
from model.nssage import NSSAGE
from model.sasrec import SASRec
from model.asage import ASAGE
from model.textsage_dask import TextSAGEDask
from model.fastsage import FastSAGE
from model.rgcn import RGCN
from model.mrec import MRec
from trainer import Trainer

#import register
MODELS = {
    'mf': MF, #  典型的なMatrix Factorization
    'lgn': LightGCN,
    'sage': GraphSAGE,
    'radj': rAdjGCN,
    'pinsage':PinSAGE,
    'textsage': TextSAGE,
    'textsage_id': TextSAGE_ID,
    'tgrec': TGRec,
    'tgsrec': TGSRec,
    'lightsage': LightSAGE,
    'tgrec2': TGRec2,
    'pinsage': PinSAGE,
    'fsage': FSAGE,
    'rsage': RSAGE,
    'sasgnn': SASGNN,
    'gnn': GNN,
    'nssage': NSSAGE,
    'sasrec': SASRec,
    'asage': ASAGE,
    'dask': TextSAGEDask,
    'fastsage': FastSAGE,
    'rgcn': RGCN,
    'mrec': MRec
}

if __name__=='__main__':
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    dataset = Loader()
    print(dataset.n_users)
    print(dataset.trainDataSize)
    model = MODELS[world.model_name](world.config, dataset)
    trainer = Trainer(world.config, dataset, model)
    trainer.train_epoch()
    