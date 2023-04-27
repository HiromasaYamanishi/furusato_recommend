import pandas as pd
class PartnerMerge:
    def __init__(self, partner_df: pd.DataFrame):
        self._partner_df = partner_df
        
    def transform(self, product_unique_df):
        product_unique_df = pd.merge(product_unique_df, self._partner_df[['partner_id', 'head_office_pref', 'head_office_addr01']], on='partner_id', how='left')
        return product_unique_df