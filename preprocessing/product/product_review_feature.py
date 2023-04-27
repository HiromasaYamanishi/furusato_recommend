import pandas as pd
from logging import getLogger 
from .utils import join_nouns

class ProductReviewFeature:
    TFIDF_THRESHOLD = 0.1
    def __init__(self, product_unique_df: pd.DataFrame, review_info: pd.DataFrame, tfidf_vec):
        self._n_product = len(product_unique_df)
        self._review_info = review_info
        self._tfidf_vec = tfidf_vec
        self._review_cnt = [0 for _ in range(self._n_product)]
        self._review_rate_total = [0 for _ in range(self._n_product)]
        self._product_review_texts = ['' for _ in range(self._n_product)]
        self._product_review_tokenized_texts = ['' for _ in range(self._n_product)]
        self.logger = getLogger(__name__)
        
        
        self._review_info['comment_tokenized'] = self._review_info['comment'].parallel_apply(join_nouns)
        self.count_review(review_info)
        self._review_tfidf_vec = self.get_tfidf_vec()
        
    def update_info(self, n_product):
        self._n_products = n_product
        
    def update_feature(self, new_review_info):
        self._review_info = pd.concat([self._review_info, new_review_info])
        self.count_review(new_review_info)
        
    def get_tfidf_vec(self, tfidf_vec):
        review_tfidf_vec = tfidf_vec.transform(pd.Series(self._product_review_tokenized_texts))
        review_tfidf_vec.data = review_tfidf_vec.data>=ProductReviewFeature.TFIDF_THRESHOLD
        review_tfidf_vec.eliminate_zeros()
        return review_tfidf_vec
        
        
    def count_review(self, review_df: pd.DataFrame):
        for (cf_product, review_rate, review_comment, review_comment_tokenized) in \
            zip(review_df['cf_product'].values, review_df['recommend_level'].values, \
                review_df['comment'].values, review_df['comment_tokenized'].values):
            if cf_product is None or pd.isna(cf_product): continue
            self._review_cnt[int(cf_product)]+=1
            self._review_rate_total[int(cf_product)]+=review_rate
            self._product_review_texts[int(cf_product)] += review_comment
            self._product_review_tokenized_texts[int(cf_product)] += review_comment_tokenized
            
        
        
        
        
        
        
        
        
        