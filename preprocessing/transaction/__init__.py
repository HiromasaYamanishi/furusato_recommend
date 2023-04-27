import pandas as pd

class TransactionInfo:
    def __init__(self, transaction_df=None, new_transaction_df=None):
        self._transaction_df = transaction_df
        self._new_transaction_df = new_transaction_df
    
    def update(self, new_transaction_df):
        # new_transaction_dfは今までのtransaction_dfと被りがないことを想定
        total_transaction_df = pd.concat([self._transaction_df, new_transaction_df])
        self._transaction_df = total_transaction_df
        self._new_transaction_df = new_transaction_df
        
    @property
    def n_transaction(self) -> int:
        return len(self._transaction_df)