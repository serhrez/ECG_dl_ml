from utils.snomed_ct import get_snomed, get_arrhythmia
from utils.functions import read_config, add_values_to_dictionary, \
    dictionary_perform, shuffle_data, under_sample_by_least, \
    save_log_np, identify_optimal_thresholds, one_hot_encode, \
    flatten_array, get_highly_correlated_features, calculate_f1_score_for_df, \
    add_leads_to_feats_list
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score as sklearn_f1_score
from dataset import Dataset, split_dataset
import importlib
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

def calc_drop_columns(feature_importances, features):
    drop_columns = []
    split_feats = np.array_split(feature_importances, 12)
    sum_importances = np.sum(split_feats, axis=0)
    feature_elim_idx = np.argmin(sum_importances)
    feature_elim_indices = np.arange(
        feature_elim_idx, 
        feature_elim_idx + (len(features) // 12) * 12, 
        len(features) // 12)
    drop_columns = [features[idx] for idx in feature_elim_indices]
    return drop_columns

def find_best_drop_columns(df, config, labels_order, highly_corr_feats):
    def get_lead_cols():
        all_cols = df.get_X_dataframe().columns
        non_leads_cols = df.get_non_leads_columns()
        # Remove non_leads_cols from all_cols
        lead_cols = [col for col in all_cols if col not in non_leads_cols]
        return lead_cols
    df_drop_columns = highly_corr_feats.copy()
    rf_drop_features = []
    df.set_drop_columns(df_drop_columns)

    f1_scores = []
    initial_f1_scores, feature_importances = calculate_f1_score_for_df(
        df, config['n_samples'], config['classifier_parameters'], 
        labels_order, num_of_splits=10)
    while len(get_lead_cols()) >= 12:
        feature_importances = np.mean(feature_importances, axis=0)
        # Remove age and is_male features
        feature_importances = feature_importances[1:-1]
        drop_columns = calc_drop_columns(feature_importances, get_lead_cols())
        df_drop_columns.extend(drop_columns)
        df.set_drop_columns(df_drop_columns)
        f1_scores_temp, feature_importances = calculate_f1_score_for_df(
            df, config['n_samples'], config['classifier_parameters'], 
            labels_order, num_of_splits=10)
        # Add the dropped feature without "_1" suffix
        rf_drop_features.append(drop_columns[0][:-2])
        f1_scores.append(f1_scores_temp)
    return initial_f1_scores, f1_scores, rf_drop_features

if __name__ == "__main__":
    log_dict = {}

    config = read_config('configs/rf_backward_stepwise.json')
    log_dict['config'] = config

    arrhythmia_label = config['arrhythmia_label']
    labels_order = [arrhythmia_label, 'NSR', 'Other']

    dataset = Dataset(config['dataset_path'], config['description_file_name'])
    dataframe = dataset.get_pandas_dataframe(arrhythmia_label)

    features_df = pd.read_csv(config['features_path'])

    ds_module = importlib.import_module(config['ds_class_module'])
    ds_class = getattr(ds_module, config['ds_class_name'])
    df = ds_class(dataframe, features_df)

    highly_corr_feats_threshold = config['highly_corr_feats_threshold']
    df.set_drop_columns([])
    highly_corr_feats = get_highly_correlated_features(df, highly_corr_feats_threshold)
    highly_corr_feats = add_leads_to_feats_list(highly_corr_feats)
    log_dict['highly_corr_feats'] = highly_corr_feats
    df.set_drop_columns(highly_corr_feats)
    initial_f1_score, rf_f1_scores, rf_drop_features = find_best_drop_columns(df, config, labels_order, highly_corr_feats)
    log_dict['initial_f1_score_mean'] = np.mean(initial_f1_score)
    log_dict['rf_f1_scores_mean'] = np.mean(rf_f1_scores, axis=1)
    log_dict['rf_drop_features'] = rf_drop_features
    log_dict['initial_f1_score'] = initial_f1_score
    log_dict['rf_f1_scores'] = rf_f1_scores
    save_log_np(log_dict, config['log_name'])