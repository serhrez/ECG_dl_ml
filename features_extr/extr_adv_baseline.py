# Add the directory of the parent folder to the system path
import sys
sys.path.append('..')

from utils.functions import read_config, read_description
import os
import scipy.io as sio
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.stats
from utils.functions import preprocess

config = read_config('configs/extract_feats_config.json')

description_path = os.path.join(config['dataset_path'], config['description_file_name'])
description = read_description(description_path)

def extract_features(hea_path, mat_path):
    # Returns: Arrhythmias, Age, Sex, are_multiple_arrhythmias, and the root-mean square
    # Open hea file
    with open(hea_path, 'r') as f:
        lines = f.readlines()
        age = lines[13][7:].strip()
        sex = lines[14][7:].strip()
        arrhythmias = lines[15][6:].strip()
        are_multiple_arrhythmias = arrhythmias.find(',') != -1
        frequency = int(lines[0].split(' ')[2])
    # Read matlab file, extract relevant features
    mat = sio.loadmat(mat_path)['val']
    # Preprocess mat
    mat = preprocess(mat, frequency)
    rms = np.sqrt(np.mean(mat**2, axis=1))
    mean = np.mean(mat, axis=1)
    std = np.std(mat, axis=1)
    range = np.ptp(mat, axis=1)
    skew = scipy.stats.skew(mat, axis=1)
    kurtosis = scipy.stats.kurtosis(mat, axis=1)

    # Create an array of age, sex, and ecg_signal features
    
    return [arrhythmias] + [age] + [sex] + [are_multiple_arrhythmias] + rms.tolist() + mean.tolist() + std.tolist() + range.tolist() + skew.tolist() + kurtosis.tolist()

record_names_set = set()
features = []

for dataset in tqdm(description.keys()):
    for arrhythmia in tqdm(description[dataset].keys(), leave=False):
        for record in tqdm(description[dataset][arrhythmia], leave=False):
            if record in record_names_set:
                continue
            hea_path = os.path.join(config['dataset_path'], record + '.hea')
            mat_path = os.path.join(config['dataset_path'], record + '.mat')
            if not os.path.exists(hea_path) or not os.path.exists(mat_path):
                continue
            feats = extract_features(hea_path, mat_path)
            record_names_set.add(record)
            features.append([dataset, record] + feats)

# Create dataframe with features and columns names
column_names = ["dataset", "record", "arrhythmias", "age", "sex", "are_multiple_arrhythmias"] + \
         [f"rms{i}" for i in range(1, 13)] + \
        [f"mean{i}" for i in range(1, 13)] + \
        [f"std{i}" for i in range(1, 13)] + \
        [f"range{i}" for i in range(1, 13)] + \
        [f"skew{i}" for i in range(1, 13)] + \
        [f"kurtosis{i}" for i in range(1, 13)]
df = pd.DataFrame(features, columns=column_names)
# Save dataframe to csv file
df.to_csv(config['extract_feats_path'], index=False)