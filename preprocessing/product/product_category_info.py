from __future__ import annotations
import pandas as pd
import numpy as np
import category_encoders as ce
from collections import defaultdict
from scipy.sparse import coo_matrix

class CategoryInfo:
    def __init__(self, product_category_df=None, new_category_df=None, encoder=None, max_category_num=0):
        # category_dfは累計のcategory_dfを指す
        # new_category_dfは新しく入ってきたdfを指す
        self._category_df = product_category_df
        self._new_category_df = new_category_df
        self._encoder = encoder
        self._max_category_num = max_category_num
        self.initialize(product_category_df)
        
    @property
    def product_category_df(self)->pd.DataFrame:
        return self._category_df
        
    def initialize(self, category_df):
        encoder = ce.OrdinalEncoder(return_df=False, handle_missing='return_nan', handle_unknown='return_nan')
        category_label = encoder.fit_transform(category_df['category_id'].astype('category'))
        max_category_num = max(category_label)
        max_category_num+= 1 #nanと新しいクラスは最大値とする
        # feature offsetをずらして全体として0インデックスにする
        category_label = np.nan_to_num(category_label, nan=max_category_num)
        category_df['category_id'] = category_label
        
        self._category_df = category_df
        self._new_category_df = category_df
        self._encoder = encoder
        self._max_category_num = max_category_num
    
    def update(self, new_category_df):
        # new_category_dfは今まで見たことのないdfと仮定
        encoder = self._encoder
        new_category_label = encoder.transform(new_category_df['category_id'])
        new_category_label = np.nan_to_num(new_category_label, nan=self._max_category_num)
        new_category_df['category_id'] = new_category_label
        category_df = pd.concat([self._category_df, new_category_df])
        
        self._category_df = category_df
        self._new_category_df = new_category_df
        
        
class ProductCategoryInfo:
    def __init__(self, product_unique_df: pd.DataFrame,  product_category_df: pd.DataFrame, n_product):
        self._product_unique_df = product_unique_df
        self._product_category_df = product_category_df
        self._product_category_set = defaultdict(set)
        
        self._n_product = n_product
        self._n_category = self._product_category_df
        
        self._encoder = ce.OrdinalEncoder(return_df=False, handle_missing='return_nan', handle_unknown='return_nan')
        self._category_label = self._encoder.fit_transform(self._product_category_df['category_id'])
        
        product_category_coo_row, product_category_coo_col = [], []
        for cf_product, category_id in zip(product_category_df['cf_product'], self._category_label):
            if pd.isna(cf_product) or pd.isna(category_id): continue
            if category_id in self._product_category_set[cf_product]:continue
            self._product_category_set[cf_product].add(category_id)
            product_category_coo_row.append(cf_product)
            product_category_coo_col.append(category_id)
            
        product_category_coo_data = [1 for _ in range(len(product_category_coo_row))]
        self._product_category_coo = coo_matrix((product_category_coo_data, (product_category_coo_row, product_category_coo_col)),
                                                shape=(self._n_product, self._n_category))
            
    def initialize(self, product_unique_df, product_category_df, productid_converter):
        product_category_set = self._product_category_set
        product_category_coo = self._product_category_coo
        
        
        product_category_coo_row,  product_category_coo_col = [], []
        print(product_category_df['product_id'], product_category_df['category_id'])
        for product_id, category_id in zip(product_category_df['product_id'], product_category_df['category_id']):
            if pd.isna(product_id) or pd.isna(category_id): continue
            if product_id is None or category_id is None: continue
            if  product_id not in productid_converter: continue
            unique_id = productid_converter[product_id]
            if category_id in product_category_set[unique_id]:continue
            product_category_set[unique_id].add(category_id)
            product_category_coo_row.append(unique_id)
            product_category_coo_col.append(category_id)
            
        #FIXME: ここよくない
        max_product_num = int(max(product_unique_df['cf_product'])) + 1
        max_category_num = int(max(product_category_df['category_id'])) + 1
        product_category_coo = coo_matrix(([0 for _ in range(len(product_category_coo_row))], (product_category_coo_row, product_category_coo_col)), shape=(max_product_num, max_category_num))
        
        return ProductCategoryInfo(product_unique_df, product_category_df, product_category_set, product_category_coo)