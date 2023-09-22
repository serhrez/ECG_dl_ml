from utils.snomed_ct import get_snomed, get_arrhythmia
from utils.functions import read_config, add_values_to_dictionary, \
    dictionary_perform, shuffle_data, under_sample_by_least, \
    save_log_np, identify_optimal_thresholds, one_hot_encode, \
    flatten_array, get_highly_correlated_features, \
    add_leads_to_feats_list, get_df
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score as sklearn_f1_score
from dataset import Dataset
import importlib
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import itertools
import random
from sklearn.utils import class_weight

def calculate_f1_score_for_df(df, df_test, n_samples, classifier_parameters, labels_order, num_of_splits=10, should_undersample=False):
    from dataset import Dataset, split_dataset
    (X, columns), Y = df.get_X(), df.get_Y(labels_order)

    def do_work(train_idx, val_idx):
        X_train, X_val = X[train_idx], X[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]
        X_train, Y_train = shuffle(X_train, Y_train, random_state=0)
        test_X = df_test.get_X()[0]
        test_Y = df_test.get_Y(labels_order)
        # Shuffle test_X, test_Y
        test_X, test_Y = shuffle(test_X, test_Y, random_state=0)
        # Undersample the minority class
        if should_undersample:
            X_train, Y_train = under_sample_by_least(X_train, Y_train)

        X_train = X_train[:n_samples]
        Y_train = Y_train[:n_samples]
        best_thresholds = [
            0.24414194034630587,
            0.4219608486304315,
            0.44315606056256696
        ]
        
        classifier = RandomForestClassifier(**classifier_parameters, class_weight='balanced')
        classifier.fit(X_train, Y_train)
        Y_val_pred_proba = classifier.predict_proba(X_val)
        Y_val_pred = (Y_val_pred_proba >= best_thresholds).astype(int)
        Y_val_pred = np.argmax(Y_val_pred, axis=1)

        Y_test_pred_proba = classifier.predict_proba(test_X)

        Y_test_pred = (Y_test_pred_proba >= best_thresholds).astype(int)
        Y_test_pred = np.argmax(Y_test_pred, axis=1)
        test_classification = classification_report(test_Y, Y_test_pred, target_names=labels_order)
        val_classification = classification_report(Y_val, Y_val_pred, target_names=labels_order)
        train_classification = classification_report(Y_train, classifier.predict(X_train), target_names=labels_order)
        print("Train:")
        print(train_classification)
        
        print("Val:")
        print(val_classification)

        print("Val_test:")
        print(test_classification)

        print("Val f1_score:")
        print(sklearn_f1_score(Y_val, Y_val_pred, average='macro'))
        print("Test f1_score:")
        print(sklearn_f1_score(test_Y, Y_test_pred, average='macro'))
        
        f1_score = sklearn_f1_score(df_test.get_Y(labels_order), Y_test_pred, average='macro')

        return test_classification, val_classification, f1_score
    
    split1 = list(split_dataset(df.get_X_dataframe(), 2))[0]

    test_classification, val_classification, f1_score = do_work(split1[0], split1[1])
    
    return test_classification, val_classification, f1_score

config = {
    "arrhythmia_label": "AF",
    "dataset_path": "D:\\research_old\\research_large_files\\card_challenge\\training",
    "description_file_name": "train_description.json",
    "features_path": "features/2020_all/whole_ds_processed.csv",
    "ds_class_module": "datahelpers.allpreddata",
    "ds_class_name": "AllPredData",
    "test_description_file_name": "test_description.json"
}

arrhythmia_label = config['arrhythmia_label']
labels_order = [arrhythmia_label, 'NSR', 'Other']

df = get_df(config, arrhythmia_label, False)
df_test = get_df(config, arrhythmia_label, True)

grid_params = {
    "n_estimators": 150,
    "min_samples_split": 2,
    "min_samples_leaf": 2,
    "max_depth": 97,
    "bootstrap": True,
    "n_samples": 40000
}

classifiers_p_list = []
test_cl_list = []
val_cl_list = []
f1_scores_cl_list = []

classifier_parameters = grid_params.copy()
del classifier_parameters['n_samples']


config['classifier_parameters'] = grid_params

test_classification, val_classification, f1_score = calculate_f1_score_for_df(
        df, df_test, grid_params['n_samples'], classifier_parameters, 
        labels_order, num_of_splits=10, should_undersample=False)

print("Test classification:")
print(test_classification)
print("Val classification:")
print(val_classification)
print("F1 score: {}".format(f1_score))
print("")