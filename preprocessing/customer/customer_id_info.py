import pandas as pd
import numpy as np
import datetime
def birth_year(birth):
    if pd.isna(birth):
        return None
    else:
        if 'AM' in birth:
            return datetime.datetime.strptime(birth, '%m/%d/%Y %H:%M:%S AM').year
        elif 'PM' in birth:
            return datetime.datetime.strptime(birth, '%m/%d/%Y %H:%M:%S PM').year
        
class CustomerIDInfo:
    '''
    CustomerIDと実験IDの対応を管理する
    CustomerIDと実際のIDは1対1対応になっているので, 連番で管理する
    '''
    def __init__(self, customer_basic_info_df):
        self._customer_ids = customer_basic_info_df['customer_id']
        
    def update(self, new_customer_basic_info_df):
        current_customer_ids = self._customer_ids
        new_customer_ids = new_customer_basic_info_df['customer_id']
        concat_customer_ids = pd.concat([current_customer_ids, new_customer_ids])
        
        self._customer_ids = concat_customer_ids
        
    @property
    def n_customer(self):
        return len(self._customer_ids)
    
    def convert_df(self, customer_df: pd.DataFrame) -> pd.DataFrame:
        customer_df['cf_customer'] = np.arange(len(customer_df))
        return customer_df
        
        
    
class TimeProcessing:
    def __init__(self, customer_df: pd.DataFrame):
        self._customer_df = customer_df
        
    def transform(self):
        customer_df = self._customer_df
        customer_df['birth_year'] = customer_df['birth_year'].progress_apply(birth_year)
        customer_df['age'] = 2023 - customer_df['birth_year']
        customer_df['age'] = customer_df['age'].clip(0, 100)
        return customer_df