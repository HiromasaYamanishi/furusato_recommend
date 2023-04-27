import logging
import pandas as pd
import numpy as np
from typing import List
from collections import defaultdict
import scipy.sparse as sparse
from scipy.sparse import coo_matrix, csr_matrix, vstack

class CustomerNumericFeature:
    class FeatureCounter:
        '''
        (ユーザー/商品)がどの属性の(商品/ユーザー)を(買った/買われた)かのカウンター
        '''
        def __init__(self, n_entity: int, col_name: str, col: pd.Series):
            '''
            ユーザーがどのカテゴリーを買ったかのカウンターを作成する場合
            n: ユーザー数
            col_name: カテゴリー
            col: productデータベースのカテゴリ-
            '''
            self._counter_name = col_name
            self._counter_height = n_entity
            self._counter_width = col.nunique() #カウンターのサイズ
            self._col = col.values
            self._classes = [c for c in col.unique() if not pd.isna(c)]
            self._classname_to_id = {v:i for i,v in enumerate(self._classes)}
            self._counter = defaultdict(int)
            print(f'Create Counter {col_name}')
            
        def update(self, new_n_entity: int, new_col: pd.Series):
            '''
            Customerの情報が更新されたときにCounterの情報も更新する.
            new_n_customerは新しいユーザー数,
            new_col はProduct dfにおいて新しく追加されたデータを想定
            '''
            self._counter_height = new_n_entity
            self._col = np.cancatenate([self._col, new_col.values])
            
        
        def increment(self, source_id: int, target_id: int):
            '''
            id: インクリメント対象の(ユーザー/アイテム)ID
            col_value: 対象のcolumnの値 (内部でクラスIDに変換)
            '''
            if  self._counter_height<=source_id: logging.warning(f'{source_id} exceeds counter height');return
            target_id = int(target_id)
            if target_id>=len(self._col): logging.warning(f'{target_id} exceeeds information range');return
            value = self._col[target_id]
            if pd.isna(value):return
            if value not in self._classname_to_id:return
            class_id = self._classname_to_id[value]
            self._counter[(source_id, class_id)] += 1
            #self.counter[id, class_id] += 1
            
        def get_result_numpy(self):
            cols, rows, values = [], [], []
            for (id,class_id), value in self._counter.items():
                rows.append(id)
                cols.append(class_id)
                values.append(value)
            result = csr_matrix((values, (rows, cols)), shape=(self._counter_height, self._counter_width))
            norm = sparse.diags(1/result.sum(axis=1).A.ravel()) + sparse.diags([1e-6 for _ in range(result.shape[0])]) #正規化項 ゼロ割を防ぐ
            normed_result = norm @ result
            result_numpy = normed_result.todense()
            return result_numpy  
        
    def __init__(self, n_customer: int, product_unique_df, col_names: List[str],):
        self._n_customer = n_customer
        self._feature_counters = {}
        for col_name in col_names:
            counter = self.FeatureCounter(n_customer, col_name, product_unique_df[col_name])
            self._feature_counters[col_name] = counter
            
        self._col_names = col_names
        
            
    def increment(self, transaction_data):
        for cf_customer, cf_product in zip(transaction_data['cf_customer'], transaction_data['cf_product']):
            for col in self._col_names:
                self._feature_counters[col].increment(cf_customer, cf_product)
            
            
    def initialize(self, transaction_data_orig):
        self.increment(transaction_data_orig)
            
    def update_counter(self, transaction_data_new):
        self.increment(transaction_data_new)
        
    def update_info(self, new_n_customer, new_product_unique_df):
        '''
        ユーザー数もしくはお礼品の情報が追加されたときにCounterの情報を更新する
        '''
        for col in self.col_names:
            self._feature_counters[col].update(new_n_customer, new_product_unique_df[col])
        
    def get_feature(self):
        results = []
        for counter in self._feature_counters.values():
            results.append(counter.get_result_numpy().astype(np.float16))
        results = np.concatenate(results, axis=1)
        return results
    