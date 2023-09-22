from utils.snomed_ct import get_snomed, get_arrhythmia
from utils.functions import read_config, add_values_to_dictionary, \
    dictionary_perform, shuffle_data, under_sample_by_least, \
    save_log_np, identify_optimal_thresholds, one_hot_encode, \
    flatten_array, calculate_f1_score_for_df, get_highly_correlated_features, \
    add_leads_to_feats_list
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score as sklearn_f1_score
from dataset import Dataset, split_dataset
import importlib
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

def calc_f1_scores_for_range(df, config, labels_order, range):
    f1_scores = []
    for threshold in tqdm(range):
        df.set_drop_columns([])
        highly_corr_feats = add_leads_to_feats_list(get_highly_correlated_features(df, threshold))
        df.set_drop_columns(highly_corr_feats)
        f1_score, _ = calculate_f1_score_for_df(df, config['n_samples'], config['classifier_parameters'], labels_order)
        f1_scores.append(f1_score)
    return f1_scores

def main(config):
    log_dict = {}
    log_dict['config'] = config

    arrhythmia_label = config['arrhythmia_label']
    labels_order = [arrhythmia_label, 'NSR', 'Other']

    dataset = Dataset(config['dataset_path'], config['description_file_name'])
    dataframe = dataset.get_pandas_dataframe(arrhythmia_label)

    features_df = pd.read_csv(config['features_path'])

    ds_module = importlib.import_module(config['ds_class_module'])
    ds_class = getattr(ds_module, config['ds_class_name'])
    df = ds_class(dataframe, features_df)

    all_feats_score, _ = calculate_f1_score_for_df(df, config['n_samples'], config['classifier_parameters'], labels_order)
    log_dict['all_feats_score'] = np.mean(all_feats_score)

    threshold_range = np.arange(config['threshold_range_start'], \
                                config['threshold_range_end'], \
                                config['threshold_range_step'])
    f1_scores = calc_f1_scores_for_range(df, config, \
                                         labels_order, threshold_range)
    df.set_drop_columns([])
    range = list(np.round(threshold_range, 8))
    f1_scores_mean = np.mean(f1_scores, axis=1)
    max_f1_score_mean_idx = np.argmax(f1_scores_mean)
    max_threshold = range[max_f1_score_mean_idx]
    highly_corr_feats = list(get_highly_correlated_features(df, max_threshold))
    all_f1_scores = f1_scores
    max_f1_score = f1_scores_mean[max_f1_score_mean_idx]
    log_dict['range'] = range
    log_dict['f1_scores_mean'] = f1_scores_mean
    log_dict['max_f1_score_mean_idx'] = max_f1_score_mean_idx
    log_dict['max_threshold'] = max_threshold
    log_dict['highly_corr_feats'] = highly_corr_feats
    log_dict['all_f1_scores'] = all_f1_scores
    log_dict['max_f1_score'] = max_f1_score
    save_log_np(log_dict, config['log_name'])

if __name__ == "__main__":
    config = read_config('configs/highly_corr_feats.json')
    main(config)