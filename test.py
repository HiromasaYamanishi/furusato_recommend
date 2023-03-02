import world
from dataloader import Loader
from trainer import Trainer
from model.MF import MF#, LightGCN
from model.lgcn import LightGCN
#import register
MODELS = {
    'mf': MF,
    'lgn': LightGCN
}

if __name__=='__main__':
    dataset = Loader()
    print(dataset.n_users)
    print(dataset.trainDataSize)
    model = MODELS[world.model_name](world.config, dataset)
    trainer = Trainer(world.config, dataset, model)
    trainer.train_epoch()
    