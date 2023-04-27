import pickle
from collections import OrderedDict, defaultdict
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from lightgbm.callback import CallbackEnv
from tqdm.auto import tqdm


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
        
def make_X(df):
    query_list = df['user'].value_counts()
    df = df.set_index(['user', 'item'])
    query_list = query_list.sort_index()
    df.sort_index(inplace=True)
    
    user_num_feature = np.load('/home/yamanishi/project/furusato_recommend/data/cb/user_numeric_feature22_1_10.npy')
    item_num_feature = np.load('/home/yamanishi/project/furusato_recommend/data/cb/product_numeric_feature22_1_10.npy')
    
    user_cat_feature = np.load('/home/yamanishi/project/furusato_recommend/data/cb/user_categorical_feature22_1_10.npy')
    item_cat_feature = np.load('/home/yamanishi/project/furusato_recommend/data/cb/item_categorical_feature22_1_10.npy')
    user_cat_X = user_cat_feature[df.index.get_level_values('user').values].astype(int)
    item_cat_X = item_cat_feature[df.index.get_level_values('item').values].astype(int)
    #
    user_num_X = user_num_feature[:, :500][df.index.get_level_values('user').values]
    item_num_X = item_num_feature[:, :500][df.index.get_level_values('item').values]
    #X= np.concatenate([item_cat_X, user_cat_X], axis=1)
    X= np.concatenate([item_cat_X, user_cat_X, user_num_X, item_num_X], axis=1)
    return df, X, query_list

if __name__=='__main__':
    user_cat_feature = np.load('/home/yamanishi/project/furusato_recommend/data/cb/user_categorical_feature22_1_10.npy')
    item_cat_feature = np.load('/home/yamanishi/project/furusato_recommend/data/cb/item_categorical_feature22_1_10.npy')
    cat_feature_num = user_cat_feature.shape[1] + item_cat_feature.shape[1]
    testUser, testItem = [], []
    train_file = '/home/yamanishi/project/furusato_recommend/data/cf/train22_1_10.txt'
    test_file = '/home/yamanishi/project/furusato_recommend/data/cf/test22_1_10.txt'

    train_items_dict = defaultdict(list)
    valid_items_dict = defaultdict(list)
    test_items_dict = defaultdict(list)
    trainUser, trainItem = [], []
    validUser, validItem = [], []
    with open(train_file) as f:
        for l in tqdm(f.readlines()):
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                items = [int(i) for i in l[1:]]
                uid = int(l[0])
                valid_len = int(len(items)*0.1/0.7)
                train_len = len(items)-valid_len
                trainUser.extend([uid]*train_len)
                trainItem.extend(items[:train_len])
                validUser.extend([uid]*valid_len)
                validItem.extend(items[train_len:])
                    
    val_items_dict = defaultdict(list)
    with open(test_file) as f:
        for l in tqdm(f.readlines()):
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                items = [int(i) for i in l[1:]]
                uid = int(l[0])
                testUser.extend([uid]*len(items))
                testItem.extend(items)
                test_items_dict[uid] = items
    pred_items_sage = torch.load('./data/graphsage_result.pt').flatten().numpy()
    user_index_sage = torch.repeat_interleave(torch.arange(pred_items_sage.shape[0]//50), 50).numpy()           
    y_sage = np.zeros_like(user_index_sage)
    pred_items_sage_new, user_index_sage_new , y_sage_new = [], [], []
    for i,(u,p) in tqdm(enumerate(zip(user_index_sage, pred_items_sage))):
        if p not in test_items_dict[u] and p not in val_items_dict[u]:
            pred_items_sage_new.append(p)
            user_index_sage_new.append(u)
            y_sage_new.append(0)
            #y_sage[i] = 1
            
    pred_items_lgn = torch.load('./data/lightgcn_result.pt').flatten().numpy()
    user_index_lgn = torch.repeat_interleave(torch.arange(pred_items_lgn.shape[0]//50), 50).numpy()           
    y_lgn = np.zeros_like(user_index_lgn)
    pred_items_lgn_new, user_index_lgn_new , y_lgn_new = [], [], []
    for i,(u,p) in tqdm(enumerate(zip(user_index_lgn, pred_items_lgn))):
        if p not in test_items_dict[u] and val_items_dict[u]:
            pred_items_lgn_new.append(p)
            user_index_lgn_new.append(u)
            y_lgn_new.append(0)
        # y_lgn[i] = 1
            
    train_user = np.concatenate([user_index_sage_new, user_index_lgn_new, np.array(trainUser)])
    train_item = np.concatenate([pred_items_sage_new, pred_items_lgn_new, np.array(trainItem)])
    train_y = np.concatenate([y_sage_new, y_lgn_new, np.ones(len(trainItem))])

    train_df = pd.DataFrame({'user': train_user,
        'item': train_item,
        'buy': train_y}).astype(int)

    train_df = train_df.drop_duplicates(keep='last')
    print(len(train_df))
    train_y = train_df['buy'].values

    train_df, train_X, train_query_list = make_X(train_df)
    print('process train done')

    test_user_index = validUser
    test_item_index = validItem
    test_y = np.ones(len(validUser))
    test_user_neg = np.arange(max(validUser)+1).repeat(10)
    test_item_neg = np.random.randint(0, max(testItem), len(test_user_neg))
    neg_y = np.zeros(len(test_user_neg))
    test_user_index = np.concatenate([test_user_index, test_user_neg])
    test_item_index = np.concatenate([test_item_index, test_item_neg])
    test_y = np.concatenate([test_y, neg_y])

    test_df = pd.DataFrame({'user': test_user_index,
        'item': test_item_index,
        'buy': test_y}).astype(int)

    test_df, test_X, test_query_list = make_X(test_df)
    print('process test done')

    '''
    lgb_params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "n_estimators":2000,
        "boosting_type": "gbdt",
        "num_leaves":31,
        "learning_rate":0.01,
        "importance_type": "gain",
        "random_state": 42,
    }
    lgb_fit_params = {
        "eval_metric":"ndcg",
        "eval_at":(1,2,3),
        "early_stopping_rounds": 50,
        "verbose": 10,
        "categorical_feature": categorical_cols,
    }
    '''

    model = lgb.LGBMRanker(n_estimators=1000, random_state=0)
    print('start fitting model')
    model.fit(
        train_X,
        train_y,
        group=train_query_list,
        callbacks=[
        LgbmProgressBarCallback(description="Model A"),
        ],
        categorical_feature=[i for i in range(cat_feature_num)],
        eval_set=[(test_X, test_y)],
        eval_group=[list(test_query_list)]
    )

    with open('./data/lightgbm.pkl', 'wb') as f:
        pickle.dump(model, f)