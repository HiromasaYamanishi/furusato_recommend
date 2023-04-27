import numpy as np
import pandas as pd
import category_encoders as ce

class CustomerCategoricalFeature:
    # このクラスはcustomerのカテゴリカル変数の管理を目的とする
    #
    def __init__(self, customer_unique_df=None, 
                 category_columns=['sexuality', 'yahoo_flg', 'credit_flg', 'pref', 'zip01',], 
                 categorical_features=[],
                 ):
        self._customer_unique_df = customer_unique_df
        self._categorical_features = []
        self._category_columns = category_columns
        self._label_encoders = {}
        self._max_features = {}
        self.initialize(customer_unique_df)
        
    def initialize(self, customer_unique_df):
        categorical_features = []
        customer_df = customer_unique_df
        feature_offset = 0
        for category_column in self._category_columns:
            assert category_column in customer_df.columns, f'{category_column} not exist in customer_df'
            encoder = ce.OrdinalEncoder(return_df=False, handle_missing='return_nan', handle_unknown='return_nan')
            feature = encoder.fit_transform(customer_df[category_column].astype('category'))
            # カテゴリーにおける特徴量の最大値を求める
            max_feature_num = max(feature)
            max_feature_num += 1 #nanと新しいクラスは最大値とする
            # feature offsetをずらして全体として0インデックスにする
            feature = np.nan_to_num(feature, nan=max_feature_num)
            feature += feature_offset
            self._max_features[category_column] = max_feature_num
            self._label_encoders[category_column] = encoder
            categorical_features.append(feature)
            feature_offset += max_feature_num
        categorical_features = np.concatenate(categorical_features, axis=1)

        self._categorical_features = categorical_features
        self._customer_categorical_feature_num = feature_offset + 1
    
    def update(self, new_customer_unique_df):
        #new_prouct_unique_dfは今まで見たことのないデータを想定
        categorical_features = self._categorical_features
        customer_size = new_customer_unique_df['cf_customer'].max() + 1
        if customer_size>categorical_features.shape[0]:
            new_customer_num = customer_size - categorical_features.shape[0]
            categorical_features = np.pad(categorical_features, ((0, new_customer_num), (0, 0)), 'constant')
            
        new_categorical_features = categorical_features
        del categorical_features
        feature_offset = 0
        feature_index = new_customer_unique_df['cf_customer'].values
        for i, category_col in enumerate(self._category_columns):
            encoder = self._label_encoders[category_col]
            new_categorical_feature = encoder.transform(new_customer_unique_df[category_col])
            new_categorical_feature = np.nan_to_num(new_categorical_feature, self._max_features[category_col])
            new_categorical_feature+= feature_offset
            feature_offset += self._max_features[category_col]
            new_categorical_features[feature_index, i] = np.squeeze(new_categorical_feature)
            
        new_customer_unique_df = pd.concat([self._customer_unique_df, new_customer_unique_df], axis=0)
        
        self._customer_unique_df = new_customer_unique_df
        self._categorical_features = new_categorical_features
        
    def get_feature(self):
        return self._categorical_features
        
    @property
    def customer_categorical_feature_dim(self) -> int:
        return self._customer_categorical_feature_num
        