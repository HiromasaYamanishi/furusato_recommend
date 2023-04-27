import pickle
from collections import OrderedDict, defaultdict
from typing import Optional
import time
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from lightgbm.callback import CallbackEnv
from tqdm.auto import tqdm
from train_lgbm import make_X


class LgbmProgressBarCallback:
    description: Optional[str]
    pbar: tqdm

    def __init__(self, description: Optional[str] = None):
        self.description = description
        self.pbar = tqdm()

    def __call__(self, env: CallbackEnv):

        # 初回だけProgressBarを初期化する
        is_first_iteration: bool = env.iteration == env.begin_iteration

        if is_first_iteration:
            total: int = env.end_iteration - env.begin_iteration
            self.pbar.reset(total=total)
            self.pbar.set_description(self.description, refresh=False)

        # valid_setsの評価結果を更新
        if len(env.evaluation_result_list) > 0:
            # OrderedDictにしないと表示順がバラバラになって若干見にくい
            postfix = OrderedDict(
                [
                    (f"{entry[0]}:{entry[1]}", str(entry[2]))
                    for entry in env.evaluation_result_list
                ]
            )
            self.pbar.set_postfix(ordered_dict=postfix, refresh=False)

        # 進捗を1進める
        self.pbar.update(1)
        self.pbar.refresh()

valUser, valItem = [], []
testUser, testItem = [], []
test_file = '/home/yamanishi/project/furusato_recommend/data/cf/test22_1_10.txt'
val_items_dict = defaultdict(list)
with open(test_file) as f:
    for l in tqdm(f.readlines()):
        if len(l) > 0:
            l = l.strip('\n').split(' ')
            items = [int(i) for i in l[1:]]
            val_len =  int(len(items)//3)
            test_len = len(items)-val_len
            val_items = items[:val_len]
            test_items = items[val_len:]
            uid = int(l[0])
            valUser.extend([uid]*val_len)
            testUser.extend([uid]*test_len)
            valItem.extend(val_items)
            testItem.extend(test_items)
            val_items_dict[uid] = val_items
            
k=10
pred_items_sage = torch.load('./data/graphsage_result.pt')[:, :k].flatten().numpy()
user_index_sage = torch.repeat_interleave(torch.arange(pred_items_sage.shape[0]//k), k).numpy()           
y_sage = np.zeros_like(user_index_sage)
for i,(u,p) in tqdm(enumerate(zip(user_index_sage, pred_items_sage))):
    if p in val_items_dict[u]:
        y_sage[i] = 1
pred_items_lgn = torch.load('./data/lightgcn_result.pt')[:, :k].flatten().numpy()
user_index_lgn = torch.repeat_interleave(torch.arange(pred_items_lgn.shape[0]//k), k).numpy()           
y_lgn = np.zeros_like(user_index_lgn)
for i,(u,p) in tqdm(enumerate(zip(user_index_lgn, pred_items_lgn))):
    if p in val_items_dict[u]:
        y_lgn[i] = 1
        

pred_user = np.concatenate([user_index_sage, user_index_lgn])
pred_item = np.concatenate([pred_items_sage, pred_items_lgn])
#train_y = np.concatenate([y_sage, y_lgn, np.ones(len(valItem))])

pred_df = pd.DataFrame({
    'user': pred_user,
    'item': pred_item
})
pred_df = pred_df.drop_duplicates(keep='last')
pred_result_user = pred_df['user']
pred_result_item = pred_df['item']
pred_df, pred_X, pred_query_list = make_X(pred_df)
print(pred_X.shape)
load_start = time.time()
with open('./data/lightgbm.pkl', 'rb') as f:
    model = pickle.load(f)
print('load time:', time.time()-load_start)
print('loaded model')
pred = model.predict(pred_X)
print(pred)
print(pred.shape)
pred_result_df = pd.DataFrame({'user': pred_result_user,
                  'item': pred_result_item,
                  'pred_score': pred})

pred_product_ids = defaultdict(list)
k = 10
for user, item, p in tqdm(zip(pred_result_user, pred_result_item, pred)):
    pred_product_ids[user].append((item, p))
    
for u in tqdm(pred_product_ids.keys()):
    pred_product_ids[u] = sorted(pred_product_ids[u], key=lambda x: -x[1])[:10]
    
pred_product_ids = {u: [l[0] for l in pred_product_ids[u]] for u in pred_product_ids.keys()}
with open('./data/lgbm_result.pkl', 'wb') as f:
    pickle.dump(pred_product_ids, f)