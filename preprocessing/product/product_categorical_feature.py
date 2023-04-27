import numpy as np
import pandas as pd
import category_encoders as ce
#FIXME: テスト完了   
class ProductCategoricalFeature:
    # このクラスはProductのカテゴリカル変数の管理を目的とする
    #
    def __init__(self, product_unique_df=None, 
                 category_columns=['head_office_pref', 'head_office_addr01',], 
                 categorical_features=[],
                 label_encoders = {},
                 max_features = {}):
        self._product_unique_df = product_unique_df
        self._categorical_features = categorical_features
        self._category_columns = category_columns
        self._label_encoders = label_encoders
        self._max_features = max_features
        self.initialize(product_unique_df)
        
    def initialize(self, product_unique_df):
        categorical_features = []
        product_df = product_unique_df
        feature_offset = 0
        for category_column in self._category_columns:
            assert category_column in product_df.columns, f'{category_column} not exist in product_df'
            encoder = ce.OrdinalEncoder(return_df=False, handle_missing='return_nan', handle_unknown='return_nan')
            feature = encoder.fit_transform(product_df[category_column].astype('category'))
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
    
    def update(self, new_product_unique_df):
        #new_prouct_unique_dfは今まで見たことのないデータを想定
        categorical_features = self._categorical_features
        product_size = new_product_unique_df['cf_product'].max() + 1
        if product_size>categorical_features.shape[0]:
            new_product_num = product_size - categorical_features.shape[0]
            categorical_features = np.pad(categorical_features, ((0, new_product_num), (0, 0)), 'constant')
            
        new_categorical_features = categorical_features
        del categorical_features
        feature_offset = 0
        feature_index = new_product_unique_df['cf_product'].values
        for i, category_col in enumerate(self._category_columns):
            encoder = self._label_encoders[category_col]
            new_categorical_feature = encoder.transform(new_product_unique_df[category_col])
            new_categorical_feature+= feature_offset
            feature_offset += self._max_features[category_col]
            new_categorical_features[feature_index, i] = np.squeeze(new_categorical_feature)
            
        new_product_unique_df = pd.concat([self._product_unique_df, new_product_unique_df], axis=0)
        
        self._product_unique_df = new_product_unique_df
        self._categorical_features = new_categorical_features