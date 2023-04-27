import torch

import world
from dataloader import Loader
from model.graphsage import GraphSAGE
from model.lgcn import LightGCN
from model.MF import MF  # , LightGCN
from model.pinsage import PinSAGE
from model.radj import rAdjGCN
from model.textsage import TextSAGE
from trainer import Trainer

#import register
MODELS = {
    'mf': MF,
    'lgn': LightGCN,
    'sage': GraphSAGE,
    'radj': rAdjGCN,
    'pinsage':PinSAGE,
    'textsage': TextSAGE
}

if __name__=='__main__':
    dataset = Loader()
    print(dataset.n_users)
    print(dataset.trainDataSize)
    model = MODELS[world.model_name](world.config, dataset)
    if world.model_name=='textsage':
        model.load_state_dict(torch.load('/home/yamanishi/project/furusato_recommend/checkpoints/textsage/64_2_0.6_22_1_10.pth'))
    elif world.model_name=='lgn':
        model.load_state_dict(torch.load('/home/yamanishi/project/furusato_recommend/checkpoints/lgn/64_3_0.6_22_1_10.pth'))
    model.eval()
    trainer = Trainer(world.config, dataset, model)
    #results = trainer.test()
    rating_list = trainer.get_topk_list(k=50)
    rating_list = torch.cat(rating_list, axis=0)
    if world.model_name=='textsage':
        torch.save(rating_list, './data/graphsage_result.pt')
    elif world.model_name=='lgn':
        torch.save(rating_list, 'data/lightgcn_result.pt')
    print(rating_list)
    