import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
for _ in range(2):
    current = os.path.dirname(current)
sys.path.append(current)

from utils.functions import get_all_records
import pandas as pd
from datahelpers.dldata import DLDatasetNoBatch
import numpy as np
import pandas as pd
from tqdm import tqdm

def get_whole_dataset(config, lead):
    records_names = get_all_records(config, config['whole_description_file_name'])
    records_names = records_names
    whole_df = pd.DataFrame({
        'Record': records_names,
        'Arrhythmia': ['Dummy'] * len(records_names)
    })
    dl_dataset = DLDatasetNoBatch(whole_df, config['dataset_path'], config['record_len'])
    dl_dataset.set_transform_y(lambda _: [0, 0, 0])
    dl_dataset.set_leads([lead])
    data = np.array([dl_dataset[i][0] for i in tqdm(range(len(dl_dataset)), desc='get_whole_dataset', leave=False)])
    return data, records_names
