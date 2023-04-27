from __future__ import annotations
import pandas as pd
import pandarallel
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
from scipy.sparse import vstack, csr_matrix
from janome.analyzer import Analyzer
from janome.charfilter import (RegexReplaceCharFilter,
                               UnicodeNormalizeCharFilter)
from janome.tokenfilter import (CompoundNounFilter, ExtractAttributeFilter,
                                LowerCaseFilter, POSKeepFilter,
                                TokenCountFilter)
from typing import List, Tuple, Dict
import logging
from sentence_transformers import SentenceTransformer
from .utils import join_nouns

class ProductTextFeature:
        
        
    def __init__(self, product_unique_df: pd.DataFrame):
        self._text_cols = ['name', 'main_comment', 'main_list_comment']
        self._tokenized_text_cols = ['name_tokenized', 'main_comment_tokenized', 'main_list_comment_tokenized']
        self._sentence_transformer = SentenceTransformer('stsb-xlm-r-multilingual')
        
        product_unique_df = ProductTextFeature.fillna(product_unique_df, cols=self._text_cols)
        product_unique_df['all'] = product_unique_df['name'] + product_unique_df['main_comment'] + product_unique_df['main_list_comment']
        product_unique_df = ProductTextFeature.tokenize(product_unique_df, ['all'] + self._text_cols)
        product_unique_df = ProductTextFeature.fillna(product_unique_df, cols=['all_tokenized'] + self._tokenized_text_cols)       
        tfidf_vec = TfidfVectorizer(max_df=0.5, min_df=3, max_features=50000)
        tfidf_vec.fit(product_unique_df['all_tokenized'])
        name_vec, main_comment_vec, main_list_comment_vec = \
            ProductTextFeature.get_text_vec(product_unique_df, tfidf_vec, self._tokenized_text_cols)
            
        sentence_embedding = self._sentence_transformer.encode(product_unique_df['all'].values, batch_size=1000)
       
       
        self._tfidf_vec = tfidf_vec
        self._name_vec = name_vec
        self._main_comment_vec = main_comment_vec
        self._main_list_comment_vec = main_list_comment_vec
        self._name_tokenized = product_unique_df['name_tokenized'].values
        self._main_comment_tokenized = product_unique_df['main_comment_tokenized'].values
        self._main_list_comment_tokenized = product_unique_df['main_list_comment_tokenized'].values
        self._sentence_embedding = sentence_embedding
        
    def update(self, new_product_unique_df: pd.DataFrame) -> None:
        
        new_product_unique_df['all'] = new_product_unique_df['name'] + new_product_unique_df['main_comment'] + new_product_unique_df['main_list_comment']
        new_product_unique_df = ProductTextFeature.fillna(new_product_unique_df, cols=['all'] + self._text_cols)
        new_product_unique_df = ProductTextFeature.tokenize(new_product_unique_df, self._text_cols)
        new_product_unique_df = ProductTextFeature.fillna(new_product_unique_df, cols=self._tokenized_text_cols)   
        
        tfidf_vec = self._tfidf_vec
        new_name_vec, new_main_comment_vec, new_main_list_comment_vec =\
            ProductTextFeature.get_text_vec(new_product_unique_df,  tfidf_vec, tokenized_cols=self._tokenized_text_cols)
        new_sentence_embedding = self._sentence_transformer.encode(new_product_unique_df['all'].values, batch_size=1000)
            
        self._name_vec= vstack([self._name_vec, new_name_vec])
        self._main_comment_vec = vstack([self._main_comment_vec, new_main_comment_vec])
        self._main_list_comment_vec = vstack([self._main_list_comment_vec, new_main_list_comment_vec])
        self._name_tokenized = np.concatenate([self._name_tokenized, new_product_unique_df['name_tokenized'].values])
        self._main_comment_tokenized = np.concatenate([self._main_comment_tokenized, new_product_unique_df['main_comment_tokenized'].values])   
        self._main_list_comment_tokenized = np.concatenate([self._main_list_comment_tokenized, new_product_unique_df['main_list_comment_tokenized'].values])   
        self._sentence_embedding = np.concatenate([self._sentence_embedding, new_sentence_embedding])
        
    @staticmethod
    def get_text_vec(product_unique_df: pd.DataFrame, tfidf_vec: TfidfVectorizer, tokenized_cols: List[str])-> Tuple[csr_matrix]:
        all_text_vecs = []
        for col in tokenized_cols:
            text_vec = tfidf_vec.transform(product_unique_df[col])
            all_text_vecs.append(text_vec)
        return tuple(all_text_vecs)
            
        
    @staticmethod
    def tokenize(product_unique_df: pd.DataFrame, cols: List[str])-> pd.DataFrame:
        for col in cols:
            product_unique_df[f'{col}_tokenized'] = product_unique_df[col].parallel_apply(join_nouns)
        return product_unique_df
        
    @staticmethod
    def fillna(product_unique_df: pd.DataFrame, cols: List[str]):
        '''
        DataFrameの指定した行をfillnaする
        '''
        for col in cols:
            product_unique_df[col] = product_unique_df[col].fillna(' ')
            
        #product_unique_df['name'] = product_unique_df['name'].fillna(' ')
        #product_unique_df['main_comment'] = product_unique_df['main_comment'].fillna(' ')
        #product_unique_df['main_list_comment'] = product_unique_df['main_list_comment'].fillna(' ')
        return product_unique_df

        