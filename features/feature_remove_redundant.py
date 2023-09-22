# Add the directory of the parent folder to the system path
import sys
sys.path.append('..')

import numpy as np
import pandas as pd
from time import time
from mlpython.utils.functions import read_config, save_log, add_leads_to_feats_list

config = read_config('configs/remove_redundant.json')
df = pd.read_csv(config['csv_path'])

redundant_cols = config['highly_corr_feats'] + config['rf_redundant_feats']

redundant_cols = add_leads_to_feats_list(redundant_cols)

new_df = df.drop(redundant_cols, axis=1)

new_df.to_csv(config['output_csv_path'], index=False)