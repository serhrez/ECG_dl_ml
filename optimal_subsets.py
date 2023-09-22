from utils.snomed_ct import get_snomed, get_arrhythmia
from utils.functions import read_config, add_values_to_dictionary, \
    dictionary_perform, shuffle_data, under_sample_by_least, \
    save_log_as_npy, one_hot_encode, \
    flatten_array, get_highly_correlated_features, \
    add_leads_to_feats_list, get_df, identify_optimal_thresholds
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
from utils.constants import LEAD_NAMES
from time import time   
import xgboost as xgb

IS_PARALLEL = True

def train_classifier(val_df, test_df, n_samples, \
                     classifier_parameters, labels_order, \
                     optimal_thresholds, classifier_name, num_of_splits=5):
    (X, columns), Y = val_df.get_X(), val_df.get_Y(labels_order)
    (X_test, _), Y_test = test_df.get_X(), test_df.get_Y(labels_order)
    # Split data into train and validation
    feature_importances = []

    def do_work(train_idx, val_idx):
        X_train, X_val = X[train_idx], X[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]
        X_train, Y_train = shuffle(X_train, Y_train, random_state=0)

        X_train = X_train[:n_samples]
        Y_train = Y_train[:n_samples]

        if classifier_name == 'xgboost_recursive_feature_elimination':
            classifier = xgb.XGBClassifier(**classifier_parameters, objective='multi:softprob', num_class=3)
        else:
            classifier = RandomForestClassifier(**classifier_parameters, class_weight='balanced')
        classifier.fit(X_train, Y_train)
        Y_val_pred_proba = classifier.predict_proba(X_val)
        Y_val_pred = (Y_val_pred_proba >= optimal_thresholds).astype(int)
        Y_val_pred = np.argmax(Y_val_pred, axis=1)

        Y_test_pred_proba = classifier.predict_proba(X_test)
        Y_test_pred = (Y_test_pred_proba >= optimal_thresholds).astype(int)
        Y_test_pred = np.argmax(Y_test_pred, axis=1)

        return classifier.feature_importances_, Y_val_pred_proba, Y_val, Y_val_pred, \
               Y_test_pred_proba, Y_test_pred
    
    if IS_PARALLEL:
        feature_importances, pred_probas, y_vals, Y_val_pred, \
            Y_test_pred_proba, Y_test_pred = zip(*Parallel(n_jobs=-1)(
            delayed(do_work)(train_idx, val_idx) for train_idx, val_idx in \
                split_dataset(val_df.get_X_dataframe(), num_of_splits)
            ))
    else:
        feature_importances, pred_probas, y_vals, Y_val_pred, \
            Y_test_pred_proba, Y_test_pred = [do_work(train_idx, val_idx) for train_idx, val_idx in \
                                            split_dataset(val_df.get_X_dataframe(), num_of_splits)]
    return {
        'val_feature_importances': feature_importances,
        'val_probabilities': pred_probas,
        'val_target_values': y_vals,
        'val_predictions': Y_val_pred,
        'test_probabilities': Y_test_pred_proba,
        'test_predictions': Y_test_pred,
        'test_target_values': Y_test
    }

def f1_score_for_cls_dict(classifier_dict):
    f1_scores = []
    for i in range(len(classifier_dict['val_target_values'])):
        Y_val = classifier_dict['val_target_values'][i]
        Y_val_pred = classifier_dict['val_predictions'][i]
        f1_score = sklearn_f1_score(Y_val, Y_val_pred, average='macro')
        f1_scores.append(f1_score)
    return np.mean(f1_scores)

def leads_importance_for_cls_dict(classifier_dict, df, num_of_leads):
    fi = np.mean(classifier_dict['val_feature_importances'], axis=0)
    feat_imp_range = df.get_lead_columns_range()
    fi = fi[feat_imp_range[0]:feat_imp_range[1]]
    fi = np.array_split(fi, num_of_leads)
    fi = np.mean(fi, axis=1)
    return fi

def get_lead_cols(df):
    all_cols = df.get_X_dataframe().columns
    non_leads_cols = df.get_non_leads_columns()
    # Remove non_leads_cols from all_cols
    lead_cols = [col for col in all_cols if col not in non_leads_cols]
    return lead_cols

def xgboost_recursive_feature_elimination(val_df, test_df, config, labels_order):
    return recursive_feature_elimination_rf(val_df, test_df, config, labels_order)
    
def recursive_feature_elimination_rf(val_df, test_df, config, labels_order):
    all_leads_names = LEAD_NAMES.copy()

    lead_feature_importances = []
    lead_steps = []
    f1_scores = []
    drop_cols = []
    classifier_dicts = []

    for lead_iter in tqdm(range(12)):
        lead_cols = get_lead_cols(val_df)
        # Split lead_cols into (12 - lead_iter) groups
        split_lead_cols = np.array_split(lead_cols, 12 - lead_iter)
        # Calculate the mean f1_score and feature_importances for each group
        classifier_dict = train_classifier(
            val_df, test_df, config['n_samples'], config['classifier_parameters'], 
            labels_order, config['optimal_thresholds'], config['method'], num_of_splits=10)
        lead_importances = leads_importance_for_cls_dict(classifier_dict, val_df, 12 - lead_iter)
        lowest_feat_imp_idx = np.argmin(lead_importances)

        classifier_dicts.append(classifier_dict)
        f1_scores.append(f1_score_for_cls_dict(classifier_dict))
        lead_feature_importances.append(lead_importances)
        lead_steps.append(all_leads_names.copy())
        drop_cols.append(split_lead_cols[lowest_feat_imp_idx].tolist())

        all_leads_names.pop(lowest_feat_imp_idx)
        val_df.set_drop_columns(flatten_array(drop_cols))
        test_df.set_drop_columns(flatten_array(drop_cols))
    report = {
        'classifier_dicts': classifier_dicts,
        'f1_scores': f1_scores,
        'lead_feature_importances': lead_feature_importances,
        'lead_steps': lead_steps,
        'drop_cols': drop_cols
    }
    return report

def main(config):
    log_dict = {}
    start_time = time()

    log_dict['config'] = config

    arrhythmia_label = config['arrhythmia_label']
    labels_order = [arrhythmia_label, 'NSR', 'Other']

    val_df = get_df(config, arrhythmia_label)
    test_df = get_df(config, arrhythmia_label, True)
    
    if config['method'] == 'recursive_feature_elimination_rf':
        report = recursive_feature_elimination_rf(val_df, test_df, config, labels_order)
    if config['method'] == 'xgboost_recursive_feature_elimination':
        report = xgboost_recursive_feature_elimination(val_df, test_df, config, labels_order)
    end_time = time()
    log_dict['report'] = report
    log_dict['duration'] = end_time - start_time
    save_log_as_npy(log_dict, config['log_name'])

if __name__ == "__main__":
    config = read_config('configs/optimal_subsets.json')
    main(config)