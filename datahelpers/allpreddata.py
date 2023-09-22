import numpy as np
import pandas as pd

class AllPredData:
    def __init__(self, dataframe, features_df):
        self.dataframe = dataframe
        self.features_df = features_df
        self.drop_columns = []

    def get_X(self):
        df = self.get_X_dataframe()
        feats = df.drop(['dataset', 'record', \
                    'arrhythmias', 'are_multiple_arrhythmias'] + list(self.dataframe.columns), axis=1)

        return np.array(feats), feats.columns
    
    def get_X_dataframe(self):
        df = self.dataframe.merge(
            self.features_df, how='inner', 
            left_on='Record', right_on='record')
        df.drop(self.drop_columns, axis=1, inplace=True)
        return df
    
    def drop_recs_with_multiple_arrhythmias(self):
        self.dataframe = self.dataframe[self.dataframe['IsMultipleArrhythmias'] == False]
        
    def set_drop_columns(self, drop_columns):
        self.drop_columns = drop_columns

    def get_Y(self, labels_order):
        df = pd.get_dummies(self.get_X_dataframe(), columns=['Arrhythmia'])
        labels = [f'Arrhythmia_{label}' for label in labels_order]
        return np.argmax(np.array(df[labels]), axis=1)
    
    def get_non_leads_columns(self):
        return list(self.get_X_dataframe().columns[:9]) + ['is_male']
    
    def get_lead_columns_range(self):
        # Age in the beginning and is_male at the end
        return [1, -1]
