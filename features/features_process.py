# Add the directory of the parent folder to the system path
import sys
sys.path.append('..')

import numpy as np
import pandas as pd
from time import time
from mlpython.utils.functions import read_config, save_log

log_dict = {}
config = read_config('configs/features_process.json')
df = pd.read_csv(config['path'])
df.replace([np.inf, -np.inf], np.nan, inplace=True)
# Replace age that is not in range [1, 100] with NaN
df.loc[~df['age'].between(1, 100), 'age'] = np.nan
log_dict['original_shape'] = df.shape
log_dict['original_columns'] = df.columns.tolist()
# Count number of NaN values in each column
sorted_null_counts = df.isnull().sum().sort_index(ascending=True)
# Display columns and number of missing values if
# the number of missing values is greater than 10% of the total number of rows
# And do not truncate the output temporarily for this line of code
NULL_COUNTS_THRESHOLD = 0.05
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(sorted_null_counts[sorted_null_counts > len(df) * NULL_COUNTS_THRESHOLD])

# Drop columns that have the number of missing values greater than 10% of the total number of rows
to_drop_columns = set()
for col in sorted_null_counts[sorted_null_counts > len(df) * NULL_COUNTS_THRESHOLD].index:
    drop_col = '_'.join(col.split('_')[:-1])
    all_drop_col = [f"{drop_col}_{i}" for i in range(1, 13)]
    to_drop_columns.update(all_drop_col)
log_dict['dropped_columns'] = list(to_drop_columns)
log_dict['null_counts_threshold'] = NULL_COUNTS_THRESHOLD

df.drop(to_drop_columns, axis=1, inplace=True)

# Remove rows that have a lot of missing values
dropped = df.dropna(thresh=len(df.columns) * 0.95)
dropped_csv_file_path = 'all_logs_related/dropped_rows.csv'
log_dict['dropped_csv_file_path'] = dropped_csv_file_path
log_dict['new_shape_after_drop'] = dropped.shape
df[~df.index.isin(dropped.index)].to_csv(dropped_csv_file_path)
df = dropped

# Impute remaining missing values

missing_vals = df.isnull().sum()
missing_vals = missing_vals[missing_vals > 0]

# Get categorical/numerical missing values columns
categorical_missing_vals = []
numerical_missing_vals = []

for col in missing_vals.index:
    if df[col].dtype == 'object' or df[col].dtype == 'bool':
        categorical_missing_vals.append(col)
    else:
        numerical_missing_vals.append(col)

log_dict['categorical_missing_vals'] = categorical_missing_vals
log_dict['numerical_missing_vals'] = numerical_missing_vals
# Impute missing values with mean of numeric column
df.fillna(df[numerical_missing_vals].mean(), inplace=True)
# Impute missing values with mode of categorical column
df.fillna(df[categorical_missing_vals].mode().iloc[0], inplace=True)
# Min-max normalize age column
df['age'] = (df['age'] - df['age'].min()) / (df['age'].max() - df['age'].min())
# Create column is_male
df['is_male'] = (df['sex'] == 'Male').astype(int)
df.drop(['sex'], axis=1, inplace=True)
df['arrhythmias'] = df['arrhythmias'].astype(str)
# Save the processed data to a new csv file
df.to_csv(config['path_processed'], index=False)
# Save log_dict to a json file
save_log(log_dict, config['log_name'], str(int(time())))
print("")