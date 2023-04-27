from collections import defaultdict
from typing import Dict, List
import pandas as pd
import Levenshtein
import numpy as np
from tqdm import tqdm
from typing import Union
import logging


#FIXME:テスト完了
class ProductIDInfo:
    '''
    ProductIDと実験に用いるIDの変換を行う
    '''
    def __init__(self, product_basic_info_df: pd.DataFrame=None, 
                 productname_remap: Dict=defaultdict(int), 
                 parentid_remap: Dict=defaultdict(int), 
                 remapped_ids=None,
                 new_info_df=None,
                 new_remapped_ids = None,
                 previous_max_id=0
                ):
        #
        #basic_info_dfは1番最初に与えるdfを想定 [n_product, columns]
        #productname_remapはproduct名をremap_idに変換 Dict[str:int]
        #parentid_remapはparetidをremap_idに変換 Dict[int:int]
        #remapped_ids は変換後のID [n_product]
        #new_basic_info_dfは新しく追加された差分
        #new_idsは新しく追加された差分のremapped_ids
        self._basic_info_df = product_basic_info_df
        self._productname_remap = productname_remap
        self._parentid_remap = parentid_remap
        self._remapped_ids = remapped_ids
        self._new_basic_info_df = new_info_df
        self._new_remapped_ids = new_remapped_ids
        self._previous_max_id = previous_max_id
        
        self.initialize(product_basic_info_df)
        
    @property
    def n_product(self) -> int:
        return max(self._remapped_ids) + 1
    
    @property
    def basic_info(self)->pd.DataFrame:
        return self._basic_info_df
    
    @property
    def max_remapped_id(self)->int:
        if self._remapped_ids is None:
            return 0
        else:
            return max(self._remapped_ids)
        
    @property
    def experiment_df(self)->pd.DataFrame:
        experiment_df = self._basic_info_df.copy() #SettingWithCopyWarningを消すためにcopyをとる
        experiment_id = self._remapped_ids
        assert len(experiment_df) == len(experiment_id)
        experiment_df.loc[:, 'cf_product'] = experiment_id
        experiment_df = experiment_df.drop_duplicates(subset='cf_product', keep='last').set_index('cf_product', drop=False)
        return experiment_df  
    
    @property
    def productid_converter(self)->Dict:
        assert len(self._remapped_ids) == len(self._basic_info_df), 'length of cf_products is different from basic_info_df'
        return {product_id: cf_product for product_id, cf_product in zip(self._basic_info_df['product_id'], self._remapped_ids)}
    
    def convert_df(self, df: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        df['cf_product'] = df['product_id'].parallel_apply(self.convert_product_id)
        return df
        
    def convert_product_id(self, product_id)-> Union[int, None]:
        converter = self.productid_converter
        if product_id not in converter: 
            #logging.warning(f'product_id {product_id} not in product_id converter')
            return None
        else:
            return converter[product_id]
    
    def get_new_experiment_df(self, unseen=False)->pd.DataFrame:
        # 最も最近追加されたDataFrameに相当するunique_dfを返す
        experiment_df = self.experiment_df
        new_unique_ids = np.unique(self._new_remapped_ids)
        new_experiment_df = experiment_df.loc[new_unique_ids, :]
        if unseen:
            unseen_df = new_experiment_df[new_experiment_df['cf_product']>self._previous_max_id]
            return unseen_df
        else:
            return new_experiment_df
              
    
    def update(self, new_product_info_df: pd.DataFrame)-> np.array:
        '''
        新しいデータフレームが渡されたときに新しデータの実験IDを生成
        今までのデータと統合し新しいデータへと更新する
        ''' 
        basic_info = self.basic_info
        original_len = len(basic_info)
        max_remap_id = self.max_remapped_id
        assert len(self._remapped_ids)>0
        assert len(self._productname_remap)>0
        assert len(self._parentid_remap)>0
        assert max_remap_id > 0
        product_names, prices, parent_ids = \
            new_product_info_df['name'].values, \
            new_product_info_df['minimum_donation_price'].values, \
            new_product_info_df['parent_product_id'].values 
            

        productname_remap, parentid_remap = self._productname_remap, self._parentid_remap
        #新しいデータの変換後のidを生成
        new_remapped_ids, new_productname_remap, new_parentid_remap \
            = ProductIDInfo.convert_productid_to_experiment_id(
                product_names, prices, parent_ids, 
                productname_remap, parentid_remap,
                max_remap_id
            )
        basic_info = pd.concat([basic_info, new_product_info_df])
        current_remapped_ids = self._remapped_ids
        remapped_ids = np.concatenate([current_remapped_ids, new_remapped_ids])
        
        new_basic_info= basic_info[original_len:]
        
        self._basic_info_df = basic_info
        self._productname_remap = new_productname_remap
        self._parentid_remap = new_parentid_remap
        self._remapped_ids = remapped_ids
        self._new_basic_info_df = new_basic_info
        self._new_remapped_ids = new_remapped_ids
        self._previous_max_id = max_remap_id
        
    def initialize(self, basic_info):
        '''
        初期段階で元のproduct_dfを渡したときに変換パターンと実験IDを生成する
        '''
        max_remap_id = self.max_remapped_id
        assert max_remap_id == 0
        product_names, prices, parent_ids = \
            basic_info['name'].values, \
            basic_info['minimum_donation_price'].values, \
            basic_info['parent_product_id'].values
        productname_remap, parentid_remap = self._productname_remap, self._parentid_remap
        remapped_ids, new_productname_remap, new_parentid_remap \
            = ProductIDInfo.convert_productid_to_experiment_id(
                product_names, prices, parent_ids, 
                productname_remap, parentid_remap,
                max_remap_id
                )
            
        self._remapped_ids = remapped_ids
        self._productname_remap = new_productname_remap
        self._parentid_remap = new_parentid_remap
        
    @staticmethod
    def convert_productid_to_experiment_id(product_names: np.array, 
                                           prices: np.array, 
                                           parent_ids:np.array, 
                                           productname_remap: Dict, 
                                           parentid_remap: Dict, 
                                           max_remapped_id: int) -> np.array:
        '''
        お礼品の名前, 値段, parent_id, 変換パターンを入力として変換後のidと新しい変換パターンを生成する
        '''
        remapped_ids = np.zeros(len(product_names))
        remapped_ids[0] = max_remapped_id
        for i,(name1, name2, price1, price2, ppi) in tqdm(enumerate(zip(product_names[:-1], product_names[1:], prices[:-1], prices[1:], parent_ids[1:]))):
            #同じ名前の製品がある場合は同じremap_idにする
            if name2 in productname_remap:
                remapped_ids[i+1] = productname_remap[name2]
            #Parent_idが存在する場合
            elif isinstance(ppi, float) and not pd.isna(ppi):
                #parent idが同じ製品がある場合は同じremap_idにする
                if ppi in parentid_remap:
                    remapped_ids[i+1] = parentid_remap[ppi]
                #名前の類似度が0.9以下で値段が1000円以上違えば異なる製品とみなす
                else:
                    if Levenshtein.ratio(name1, name2)<0.9 or abs(price1-price2)>1000:
                        max_remapped_id+=1
                    parentid_remap[ppi] = max_remapped_id
                    remapped_ids[i+1] = max_remapped_id
            #Parent_idが存在しない場合   
            else:
                if Levenshtein.ratio(name1, name2)<0.9 or abs(price1-price2)>1000:
                    max_remapped_id+=1
                productname_remap[name2] = max_remapped_id
                remapped_ids[i+1] = max_remapped_id
            remapped_ids = remapped_ids
                
        remapped_ids = remapped_ids.astype(int)
        return remapped_ids, productname_remap, parentid_remap, 
    
    
        
        
        
        