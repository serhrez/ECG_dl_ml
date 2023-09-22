# Import parent folder
import sys
sys.path.append('..')

from mlpython.utils.functions import read_config

config = read_config('make_figs/configs/baseline_char.json')

import pandas as pd
import numpy as np
from mlpython.utils.snomed_ct import get_snomed, get_arrhythmia
import scipy.stats

def calculate_feature_averages(dataframe, df_name):
    # Use numerical columns for finding mean
    numerical_columns = dataframe.select_dtypes(include=[np.number]).columns

    # Calculate the average of numerical columns
    numerical_averages = dataframe[numerical_columns].mean()

    # Calculate standard deviation of numerical columns
    numerical_std = dataframe[numerical_columns].std()

    total_df = pd.DataFrame(columns=[df_name])
    for num_col in numerical_columns:
        total_df.loc[num_col] = {df_name: f"{round(numerical_averages[num_col], 2)} ± {round(numerical_std[num_col], 2)}"}

    # Use boolean columns for finding mode
    boolean_columns = dataframe.select_dtypes(include=[bool]).columns

    boolean_true_counts = dataframe[boolean_columns].sum()
    for bool_col in boolean_columns:
        total_df.loc[bool_col] = {df_name: f"{boolean_true_counts[bool_col]} / {len(dataframe)} ({round(boolean_true_counts[bool_col] / len(dataframe) * 100, 2)}%)"}
        
    return total_df

def calculate_p_value(df1, df2, df3):
    total_df = pd.DataFrame(columns=['p-value'])
    # Use numerical columns first
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    # Calculate p-values using ANOVA between numerical columns
    for num_col in numerical_columns:
        group1 = df1[num_col].tolist()
        group2 = df2[num_col].tolist()
        group3 = df3[num_col].tolist() 
        _, p_value = scipy.stats.kruskal(group1, group2, group3)
        total_df.loc[num_col] = {'p-value': p_value}
    
    # Use boolean columns
    boolean_columns = df.select_dtypes(include=[bool]).columns
    # Calculate p-values using chi-square between boolean columns
    for bool_col in boolean_columns:
        group1 = df1[bool_col].value_counts().tolist()
        group2 = df2[bool_col].value_counts().tolist()
        group3 = df3[bool_col].value_counts().tolist()
        contingency_table = [group1, group2, group3]
        _, p_value = scipy.stats.chisquare(contingency_table)
        total_df.loc[bool_col] = {'p-value': p_value}
    
    return total_df

nsr_code = get_snomed('NSR')
arrhythmia_code = get_snomed(config['arrhythmia_label'])
df = pd.read_csv(config["features_path"])
# Add column is multiple arrhythmias
df['are_multiple_arrhythmias'] = df['arrhythmias'].str.contains(',')
df_other = df[~df['arrhythmias'].str.contains(f'{nsr_code}|{arrhythmia_code}')]
df_nsr = df[df['arrhythmias'].str.contains(nsr_code)]
df_arrhythmia = df[df['arrhythmias'].str.contains(arrhythmia_code)]

arrhythmia_avg = calculate_feature_averages(df_arrhythmia, config['arrhythmia_label'])
nsr_avg = calculate_feature_averages(df_nsr, 'NSR')
other_avg = calculate_feature_averages(df_other, 'Other')
p_value = calculate_p_value(df_arrhythmia, df_nsr, df_other)

# Combine dataframes
total_df = pd.concat([arrhythmia_avg, nsr_avg, other_avg, p_value], axis=1)

# Save dataframe
total_df.to_csv(config['save_path'])