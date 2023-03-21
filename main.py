import world
from dataloader import Loader
from trainer import Trainer
from model.MF import MF#, LightGCN
from model.lgcn import LightGCN
from model.graphsage import GraphSAGE
from model.radj import rAdjGCN
from model.pinsage import PinSAGE
from model.textsage import TextSAGE
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
    trainer = Trainer(world.config, dataset, model)
    trainer.train_epoch()
    