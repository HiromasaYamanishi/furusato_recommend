from preprocessing.product import ProductTextFeature, ProductIDInfo, ProductCategoryInfo, ProductCategoricalFeature, \
                                  ProductNumericalFeature, CategoryInfo, ProductReviewFeature
from preprocessing.customer import CustomerIDInfo, CustomerCategoricalFeature, CustomerNumericFeature
from preprocessing.transaction import TransactionInfo
from preprocessing.utils import PartnerMerge
from tqdm import tqdm
import logging
import pandas as pd
from pandarallel import pandarallel
import os

if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    pandarallel.initialize(progress_bar=True)
    tqdm.pandas()
    
    product_df = pd.read_pickle('/home/yamanishi/project/furusato_recommend/data/all/productall.pkl')
    OFFSET = 10000
    product_df_orig = product_df[:OFFSET-100]
    product_df_new = product_df[OFFSET-100:OFFSET]
    
    ### ここはProductIDInfoのテスト　###
    product_id_info = ProductIDInfo(product_df_orig)
    product_unique_df_orig = product_id_info.experiment_df
    print(len(product_unique_df_orig))
    #print(productidinfo.productid_converter)
    productid_converter = product_id_info.productid_converter
    
    
    ### ここはCategoryInfoのテスト ###
    product_category = pd.read_csv('/home/yamanishi/project/furusato_recommend/data/product_category.csv', encoding='utf16')
    CATEGORY_OFFSET=1000000
    product_category_orig = product_category[:CATEGORY_OFFSET-100]
    product_category_orig = product_id_info.convert_df(product_category_orig)
    product_category_new = product_category[CATEGORY_OFFSET-100:CATEGORY_OFFSET]
    product_category_new = product_id_info.convert_df(product_category_new)
    category_info = CategoryInfo(product_category_orig)
    category_info.update(product_category_new)
    product_category_df = category_info.product_category_df
    print('category info updated')
    '''
    ### ここはProductReviewFeatureのテスト ###
    REVIEW_OFFSET = 200000
    review_df = pd.read_csv('./data/all/review_all.csv')
    review_df_orig = review_df[:REVIEW_OFFSET-10000]
    review_df_orig = product_id_info.convert_df(review_df_orig)
    review_df_new = review_df[REVIEW_OFFSET-10000:REVIEW_OFFSET]
    review_df_new = product_id_info.convert_df(review_df_new)
    product_review_feature = ProductReviewFeature(product_unique_df_orig, review_df_orig)
    product_review_feature.update_feature(review_df_new)
    
    ### ここはProductTextFeatureのテスト　###
    product_text_feature = ProductTextFeature(product_unique_df_orig)
    print('initialize product vec', product_text_feature._name_vec.shape)
    
    product_text_feature.update(product_df_new)
    print('updated text vec', product_text_feature._name_vec.shape)
    ### ここはProductCategoryInfoのテスト ###
    product_category_info = ProductCategoryInfo()
    product_category_info = product_category_info.initialize(product_unique_df_orig, product_category_df, productid_converter)
    print('initialized product category successfully')
    '''
    ### ここはPartnerMergeとのテスト ###
    partner_df = pd.read_csv('./data/partner.csv', encoding='utf16', low_memory=False)
    partner_merger = PartnerMerge(partner_df)
    
    product_unique_df_orig = partner_merger.transform(product_unique_df_orig)
    ### ここはProductCategoricalFeatureのテスト ###
    categoricalinfo = ProductCategoricalFeature(product_unique_df_orig)
    print(categoricalinfo._categorical_features.shape)
    ### ここはProductInfoのupdateのテスト ###
    product_id_info.update(product_df_new)
    product_unique_df_new = product_id_info.get_new_experiment_df(unseen=True)
    product_unique_df_new = partner_merger.transform(product_unique_df_new)
    print(len(product_unique_df_new))
    '''
    ### ここはProductCategoricalFeatureのupdateのテスト ###
    categoricalinfo.update(product_unique_df_new)
    print(categoricalinfo._categorical_features.shape)
    '''
    ### TransactionInfoのテスト ###
    transaction = pd.read_pickle('/home/yamanishi/project/furusato_recommend/data/all/transactionall.pkl')
    print('loaded transaction')
    TRANSACTION_OFFSET = 1000000
    transaction_orig = transaction[:TRANSACTION_OFFSET-10000]
    transaction_info = TransactionInfo(transaction_orig)
    print('inititlized transaction num: ', transaction_info.n_transaction)
    assert transaction_info.n_transaction == TRANSACTION_OFFSET-10000
    transaction_new = transaction[TRANSACTION_OFFSET-10000: TRANSACTION_OFFSET]
    transaction_info.update(transaction_new)
    assert transaction_info.n_transaction == TRANSACTION_OFFSET
    print('updated transaction num:', transaction_info.n_transaction)
    
    ### CustomerInfoのテスト ###
    customer = pd.read_pickle('/home/yamanishi/project/furusato_recommend/data/all/customerall.pkl')
    CUSTOMER_OFFSET = 100000
    customer_orig = customer[:CUSTOMER_OFFSET-10000]
    customer_id_info = CustomerIDInfo(customer_orig)
    print('inititlized customer num: ', customer_id_info.n_customer)
    assert customer_id_info.n_customer==CUSTOMER_OFFSET-10000
    customer_new = customer[CUSTOMER_OFFSET-10000: CUSTOMER_OFFSET]
    customer_id_info.update(customer_new)
    assert customer_id_info.n_customer==CUSTOMER_OFFSET
    print('updated customer num:', customer_id_info.n_customer)
    
    ### CustomerCategoricalFeatureのテスト ###
    customer_categorical_feature = CustomerCategoricalFeature(customer_orig)
    customer_categorical_feature_orig = customer_categorical_feature.get_feature()
    print('orig customer categorical:', customer_categorical_feature_orig[:10], customer_categorical_feature_orig.shape)
    customer_categorical_feature.update(customer_new)
    customer_categorical_feature_new = customer_categorical_feature.get_feature()
    print('new customer categorical:', customer_categorical_feature_new[:10], customer_categorical_feature_new.shape)
    
    print('test customer category done')

    ### CustomerNumericFeatureのテスト ###
    customer_numeric_feature = CustomerNumericFeature(customer_id_info.n_customer, 
                                                        product_unique_df_orig,
                                                        col_names=['head_office_pref', 'head_office_addr01',]
                                                      )
    customer_numeric_feature.initialize(transaction_orig)
    numeric_feature_orig = customer_numeric_feature.get_feature()
    print(numeric_feature_orig)
    customer_numeric_feature.update_counter(transaction_new)
    numeric_feature_new = customer_numeric_feature.get_feature()
    print(numeric_feature_new)
    