import json
import os
import numpy as np
from scipy import signal
import logging
from time import time
from numpyencoder import NumpyEncoder
from sklearn.metrics import f1_score as sklearn_f1_score
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from joblib import Parallel, delayed
import pandas as pd
import importlib

def read_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)
    
def read_description(description_path):
    with open(description_path, 'r') as f:
        return json.load(f)
    
def save_dictionary(dictionary, file_name):
    with open(file_name, 'w') as f:
        json.dump(dictionary, f, indent=1)

dictionary_file_versions = {}

def save_dictionary(dictionary, file_name):
    with open(file_name, 'w') as f:
        json.dump(dictionary, f, indent=1)

def save_dictionary_with_file_versions(dictionary, file_name):
    PREFIX = '_v'
    global dictionary_file_versions
    if file_name in dictionary_file_versions:
        dictionary_file_versions[file_name] += 1
        with open(file_name + f'{PREFIX}{dictionary_file_versions[file_name]}.json', 'w') as f:
            json.dump(dictionary, f, indent=1)
    else:
        index = 1
        while os.path.exists(file_name + f'{PREFIX}{index}.json'):
            index += 1
        with open(file_name + f'{PREFIX}{index}.json', 'w') as f:
            json.dump(dictionary, f, indent=1)
        dictionary_file_versions[file_name] = index

def add_values_to_dictionary(to_dictionary, from_dictionary):
    for key in from_dictionary:
        if key in to_dictionary:
            if isinstance(from_dictionary[key], dict):
                add_values_to_dictionary(to_dictionary[key], from_dictionary[key])
            else:
                to_dictionary[key] += from_dictionary[key]
        else:
            if isinstance(from_dictionary[key], dict):
                new_dict = {}
                to_dictionary[key] = new_dict
                add_values_to_dictionary(new_dict, from_dictionary[key])
            else:
                to_dictionary[key] = from_dictionary[key]

def append_values_to_dictionary(to_dictionary, from_dictionary):
    for key in from_dictionary:
        if key in to_dictionary:
            if isinstance(from_dictionary[key], dict):
                append_values_to_dictionary(to_dictionary[key], from_dictionary[key])
            else:
                to_dictionary[key].append(from_dictionary[key])
        else:
            if isinstance(from_dictionary[key], dict):
                new_dict = {}
                to_dictionary[key] = new_dict
                append_values_to_dictionary(new_dict, from_dictionary[key])
            else:
                to_dictionary[key] = [from_dictionary[key]]

def append_one_val_to_dictionary(to_dictionary, key, val):
    if key in to_dictionary:
        to_dictionary[key].append(val)
    else:
        to_dictionary[key] = [val]

def dictionary_perform(dictionary, op):
    for key in dictionary:
        if isinstance(dictionary[key], dict):
            dictionary_perform(dictionary[key], op)
        else:
            dictionary[key] = op(dictionary[key])

def shuffle_data(X, Y, random_state):
    np.random.seed(random_state)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    return X[indices], Y[indices]

def under_sample_by_least(X, Y):
    unique_classes = np.unique(Y, axis=0)
    number_of_classes = {}
    for sample in Y:
        if sample in number_of_classes:
            number_of_classes[sample] += 1
        else:
            number_of_classes[sample] = 1
    min_number_of_classes = min(number_of_classes.values())
    indices = []
    number_of_classes_count = {key: min_number_of_classes for key in unique_classes}
    for i in range(len(Y)):
        if number_of_classes_count[Y[i]] > 0:
            indices.append(i)
            number_of_classes_count[Y[i]] -= 1
    return X[indices], Y[indices]

def resample_to_500hz(ecg, original_frequency):
    # resample to 500hz
    if original_frequency == float(1000):
        data = signal.resample_poly(ecg, up=1, down=2, axis=-1)  # to 500Hz
    elif original_frequency == float(500):
        data = ecg
    else:
        data = signal.resample(ecg, int(ecg.shape[1] * 500 / original_frequency), axis=1)
    return data

butter_b, butter_a  = signal.butter(3, [1 / 250, 47 / 250], 'bandpass')

def filter_signal(ecg_signal):
    # Input: ecg_signal of shape (12, n)
    # Output: ecg_signal of shape (12, n)
    return signal.filtfilt(butter_b, butter_a, ecg_signal)

def z_score_normalize(ecg_signal):
    # Input: ecg_signal of shape (12, n)
    # Output: ecg_signal of shape (12, n)
    mu = np.nanmean(ecg_signal, axis=-1, keepdims=True)
    std = np.nanstd(ecg_signal, axis=-1, keepdims=True)
    return (ecg_signal - mu) / std

def preprocess(ecg_signal, original_frequency):
    # Input: ecg_signal of shape (12, n)
    # Output: ecg_signal of shape (12, n)
    ecg_signal = ecg_signal.astype(np.float64)
    
    # Resample to 500 Hz
    ecg_signal = resample_to_500hz(ecg_signal, original_frequency)
    # 3rd order Butterworth bandpass filter with frequency band from 1 Hz to 47 Hz
    ecg_signal = filter_signal(ecg_signal)

    return ecg_signal

def safe_np_calc(arr, func):
    if len(arr) > 0:
        return func(arr)
    else:
        return 0
    
def safe_np_calc_mult(arr, funcs):
    if len(arr) > 0:
        return [func(arr) for func in funcs]
    else:
        return [0 for _ in range(len(funcs))]

def flatten_array(arr):
    flattened = []
    for sublist in arr:
        if isinstance(sublist, list):
            flattened.extend(sublist)
        else:
            flattened.append(sublist)
    return flattened

def get_logger(log_file_name):
    # Create a logger instance
    logger = logging.getLogger('logger')
    logger.setLevel(logging.ERROR)  # Set the desired logging level

    # Create a FileHandler to write log messages to a file
    file_handler = logging.FileHandler(f'errorlogs/{log_file_name}.log')
    file_handler.setLevel(logging.ERROR)  # Set the desired logging level for the file handler

    # Create a Formatter to specify the log message format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Set the formatter for the file handler
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    return logger

def save_log(dictionary, log_file_name, log_timestamp=str(int(time()))):
    with open(f'logs/{log_file_name}+{log_timestamp}.json', 'w') as f:
        json.dump(dictionary, f, indent=2)

def save_log_as_npy(dictionary, log_file_name, log_timestamp=str(int(time()))):
    with open(f'logs/{log_file_name}+{log_timestamp}.npy', 'wb') as f:
        np.save(f, dictionary)

def read_npy_log(log_file_path):
    with open(log_file_path, 'rb') as f:
        return np.load(f, allow_pickle=True).item()

def save_log_np(dictionary, log_file_name, log_timestamp=str(int(time()))):
    with open(f'logs/{log_file_name}+{log_timestamp}.json', 'w') as f:
        json.dump(dictionary, f, indent=2, cls=NumpyEncoder)

def identify_optimal_thresholds(y_true, y_pred_proba):
    num_classes = y_pred_proba.shape[1]
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_thresholds = [0.5] * num_classes  # Initialize with default threshold of 0.5 for each class
    best_f1_scores = [0.0] * num_classes

    for class_idx in range(num_classes):
        f1_scores = []
        for threshold in thresholds:
            y_pred = (y_pred_proba[:, class_idx] >= threshold).astype(int)
            f1_scores.append(sklearn_f1_score(y_true[:, class_idx], y_pred, average='binary'))
        best_thresholds[class_idx] = thresholds[np.argmax(f1_scores)]
        best_f1_scores[class_idx] = np.max(f1_scores)

    return best_thresholds, best_f1_scores

def one_hot_encode(labels):
    # Input: labels of shape (n,)
    # Output: labels of shape (n, 3)
    return np.eye(3)[labels]

def calculate_f1_score_for_df(df, n_samples, classifier_parameters, labels_order, num_of_splits=10):
    from dataset import Dataset, split_dataset
    (X, columns), Y = df.get_X(), df.get_Y(labels_order)
    # Split data into train and validation
    f1_scores = []
    feature_importances = []

    def do_work(train_idx, val_idx):
        X_train, X_val = X[train_idx], X[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]
        X_train, Y_train = shuffle(X_train, Y_train, random_state=0)

        X_train = X_train[:n_samples]
        Y_train = Y_train[:n_samples]

        classifier = RandomForestClassifier(**classifier_parameters, class_weight='balanced')
        classifier.fit(X_train, Y_train)
        Y_val_pred_proba = classifier.predict_proba(X_val)
        best_thresholds, _ = identify_optimal_thresholds(one_hot_encode(Y_val), Y_val_pred_proba)
        Y_val_pred = (Y_val_pred_proba >= best_thresholds).astype(int)
        Y_val_pred = np.argmax(Y_val_pred, axis=1)
        f1_score = sklearn_f1_score(Y_val, Y_val_pred, average='macro')
        return f1_score, classifier.feature_importances_
    
    f1_scores, feature_importances = zip(*Parallel(n_jobs=-1)(delayed(do_work)(train_idx, val_idx) for train_idx, val_idx in split_dataset(df.get_X_dataframe(), num_of_splits)))
    
    return f1_scores, feature_importances

def get_highly_correlated_features(df, threshold):
    features_df = df.get_X_dataframe()
    features_df.drop(df.get_non_leads_columns(), axis=1, inplace=True)
    assert(int(len(features_df.columns) / 12) == len(features_df.columns) / 12)
    # Split features_df into 12 groups and put them in a list
    split_df = np.array_split(features_df, 12, axis=1)

    assert(split_df[0].columns[-1][-1] == '1')
    assert(split_df[1].columns[0][-1] == '2')
    high_corr_feats = []
    for i in range(12):
        correlation_matrix = split_df[i].corr().abs()
        set_corr_feats = set()
        for j in range(len(correlation_matrix.columns)):
            for k in range(j):
                if correlation_matrix.iloc[j, k] >= threshold:
                    feature1 = '_'.join(correlation_matrix.columns[j].split('_')[:-1])
                    feature2 = '_'.join(correlation_matrix.columns[k].split('_')[:-1])
                    
                    if feature1 in set_corr_feats or feature2 in set_corr_feats:
                        continue
                    
                    set_corr_feats.add(feature1)
        high_corr_feats.append(set_corr_feats)
    high_corr_feats_sets = [set(feats) for feats in high_corr_feats]
    intersection = set.intersection(*high_corr_feats_sets)
    return intersection

def add_leads_to_feats_list(features):
    return [f'{feat}_{lead + 1}' for lead in range(12) for feat in features]

def get_dataset_df(config, arrhythmia_label, description_file_name):
    from dataset import Dataset
    dataset = Dataset(config['dataset_path'], description_file_name)
    dataframe = dataset.get_pandas_dataframe(arrhythmia_label)
    return dataframe

def get_all_records(config, description_file_name):
    from dataset import Dataset
    dataset = Dataset(config['dataset_path'], description_file_name)
    all_records = dataset.get_all_records()
    return all_records

def get_dataset_dataframe(config, arrhythmia_label, is_testing_df=False):
    from dataset import Dataset
    if is_testing_df:
        description_file_name = config['test_description_file_name']
    else:
        description_file_name = config['description_file_name']
    dataset = Dataset(config['dataset_path'], description_file_name)
    dataframe = dataset.get_pandas_dataframe(arrhythmia_label)
    return dataframe

def get_df(config, arrhythmia_label, is_testing_df=False):
    dataframe = get_dataset_dataframe(config, arrhythmia_label, is_testing_df)

    features_df = pd.read_csv(config['features_path'])

    ds_module = importlib.import_module(config['ds_class_module'])
    ds_class = getattr(ds_module, config['ds_class_name'])
    df = ds_class(dataframe, features_df)

    return df

if __name__ == "__main__":
    def resample_each(sig, orig_freq):
        return signal.resample(sig, int(len(sig) * 500 / orig_freq))
    import scipy.io as sio
    sample_ecg_signal = sio.loadmat('HR01000.mat')['val']
    sample_ecg_signal = sample_ecg_signal.astype(np.float64)
    resampled = resample_to_500hz(sample_ecg_signal, 990)
    custom_resampled = []
    for i in range(12):
        custom_resampled.append(resample_each(sample_ecg_signal[i], 990))
    custom_resampled = np.array(custom_resampled)
    # Resampled with torch audio
    from torchaudio.transforms import Resample
    import torch
    # Create torch tensor with double type
    sample_tensor = torch.tensor(sample_ecg_signal[0], dtype=torch.double)
    resample_transform = Resample(990, 500).double()
    resampled_torch = resample_transform(sample_tensor)
    import matplotlib.pyplot as plt
    # Create 3 subplots
    plt.subplot(4, 1, 1)
    plt.plot(sample_ecg_signal[0])
    plt.subplot(4, 1, 2)
    plt.plot(resampled[0])
    plt.subplot(4, 1, 3)
    plt.plot(custom_resampled[0])
    plt.subplot(4, 1, 4)
    plt.plot(resampled_torch)
    plt.show()
    print(resampled.shape)