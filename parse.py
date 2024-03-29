import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Go lightGCN")
    parser.add_argument('--bpr_batch', type=int,default=2048,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--recdim', type=int,default=64,
                        help="the embedding size of lightGCN")
    parser.add_argument('--layer', type=int,default=2,
                        help="the layer num of lightGCN")
    parser.add_argument('--lr', type=float,default=0.0001,
                        help="the learning rate")
    parser.add_argument('--decay', type=float,default=1e-7,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--dropout', type=int,default=0,
                        help="using the dropout or not")
    parser.add_argument('--keepprob', type=float,default=0.6,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--a_fold', type=int,default=1000,
                        help="the fold num used to split large adj matrix, like gowalla")
    parser.add_argument('--num_neighbors', type=int, default=5,
                        help='the number of neighbor nodes graphsage samples one time')
    parser.add_argument('--testbatch', type=int,default=10000,
                        help="the batch size of users for testing")
    parser.add_argument('--dataset', type=str,default='furusato',
                        help="available datasets: [furusato]")
    parser.add_argument('--path', type=str,default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--topks', nargs='?',default="[10,20]",
                        help="@k test list")
    parser.add_argument('--tensorboard', type=int,default=1,
                        help="enable tensorboard")
    parser.add_argument('--wandb', type=str,
                        help='wandb run name')
    parser.add_argument('--inference', type=str, default='all')
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--comment', type=str,default="lgn")
    parser.add_argument('--load', type=int,default=0)
    parser.add_argument('--epochs', type=int,default=1000)
    #parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
    parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('--model', type=str, default='lgn', help='rec-model, support [mf, lgn]')
    parser.add_argument('--train_emb', action='store_true')
    parser.add_argument('--sample_pow', type=float, default=0)
    parser.add_argument('--r', default=0.5, type=float)
    parser.add_argument('--test_span', default=10, type=int)
    parser.add_argument('--multicore', action='store_true')
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--multi_relational', type=str, default='add')
    parser.add_argument('--conv', type=str, default='gcn')
    parser.add_argument('--for_lgbm', action='store_true')
    parser.add_argument('--lgbm_ratio', type=float, default=0.1)
    parser.add_argument('--cold_start', action='store_true')
    parser.add_argument('--user_feature', type=str, default='ntw', help='user feature must in ncwt')
    parser.add_argument('--item_feature', type=str, default='ntw', help='item feature must in ncwtsr')
    parser.add_argument('--factorization', action='store_true')
    return parser.parse_args()