import logging
import pandas as pd
import numpy as np
from typing import List
from collections import defaultdict
import scipy.sparse as sparse
from scipy.sparse import coo_matrix, csr_matrix, vstack


class ProductNumericalFeature:
    def __init__(self, n_product: int, customer_unique_df: pd.DataFrame, col_names: List[str],):
        '''
        __init__時に指定した行のCounterを生成する
        '''
        self._n_product = n_product
        self._feature_counters = {}
        for col_name in col_names:
            counter = FeatureCounter(n_product, col_name, customer_unique_df[col_name])
            self._feature_counters[col_name] = counter
            
        self.col_names = col_names
        
            
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
            
            
    def initialize(self, transaction_data_orig):
        '''
        元の購買情報をカウントする
        '''
        self.increment(transaction_data_orig)
            
    def update_counter(self, transaction_data_new):
        '''
        新しい購買情報をカウントする
        '''
        self.increment(transaction_data_new)
        
    def update_info(self, new_n_product, new_customer_unique_df):
        '''
        お礼品数もしくはユーザー数が追加されたときにCounterの情報を更新する
        '''
        for col in self.col_names:
            self._feature_counters[col].update(new_n_product, new_customer_unique_df[col])
        
    def get_feature(self):
        '''
        counterの結果を結合し, 特徴量を得る.
        '''
        results = []
        for counter in self._feature_counters.values():
            results.append(counter.get_result_numpy().astype(np.float16))
        results = np.concatenate(results, axis=1)
        return results