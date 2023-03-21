import world
from dataloader import Loader
from trainer import Trainer
from model.MF import MF#, LightGCN
from model.lgcn import LightGCN
from model.graphsage import GraphSAGE
from model.radj import rAdjGCN
from model.pinsage import PinSAGE
import torch
#import register
MODELS = {
    'mf': MF,
    'lgn': LightGCN,
    'sage': GraphSAGE,
    'radj': rAdjGCN,
    'pinsage':PinSAGE
}

if __name__=='__main__':
    dataset = Loader()
    print(dataset.n_users)
    print(dataset.trainDataSize)
    model = MODELS[world.model_name](world.config, dataset)
    model.load_state_dict(torch.load('/home/yamanishi/project/furusato_recommend/checkpoints/mf/64_3.pth'))
    model.eval()
    trainer = Trainer(world.config, dataset, model)
    results = trainer.test()
    print(results)
    