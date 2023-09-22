import numpy as np
import pandas as pd

FEATURES_NAMES = ['age', 'sex'] + [f'rms{i}' for i in range(1, 12 + 1)]

class BaselinePredData:
    def __init__(self, dataframe, features_df):
        self.dataframe = dataframe
        self.features_df = features_df

    def get_X(self):
        df = self.dataframe[['Record']].merge(
            self.features_df, how='inner', 
            left_on='Record', right_on='record')
        
        feats = df[FEATURES_NAMES]
        feats = pd.get_dummies(feats, columns=['sex'])

        return np.array(feats), feats.columns

    def get_Y(self, labels_order):
        df = pd.get_dummies(self.dataframe, columns=['Arrhythmia'])
        labels = [f'Arrhythmia_{label}' for label in labels_order]
        return np.array(df[labels])