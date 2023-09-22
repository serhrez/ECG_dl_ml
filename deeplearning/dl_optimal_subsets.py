import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
for _ in range(2):
    current = os.path.dirname(current)
sys.path.append(current)

from keras.optimizers import Adam
from keras.callbacks import (ModelCheckpoint, TensorBoard, ReduceLROnPlateau,
                                        CSVLogger, EarlyStopping)
import argparse
from keras.models import Model
import os
from tqdm import tqdm
import numpy as np
import sklearn.metrics
from scipy.stats import ttest_ind
import keras.models
import gc
import time
from sklearn.utils import class_weight
from utils.snomed_ct import get_snomed
from dataset import Dataset, split_dataset
from utils.functions import read_config, get_dataset_df, get_all_records, save_log_as_npy
import datahelpers.dldata as dldata
import math
from keras.utils import Sequence
from models import get_extr_feats_model_multiple_feats_extr, model_with_needed_layers, get_feats_model
import pandas as pd
from datahelpers.dldata import DLDatasetNoBatch, DLDataset
import tensorflow as tf
from tqdm import tqdm
import random
from time import time
import json
from utils.constants import LEAD_NAMES
from models import get_feats_model
import shap
import sklearn.metrics

DIR_PATH = "deeplearning/with_feats_extr"
FEAT_NUM = 32

def get_dataset(dataframe, config, batches, 
                sequence_len, target_labels, features_dict):
    dataset = dldata.DLDataset(
        dataframe, config['dataset_path'], record_len=sequence_len)
    dataset.set_record_lookup(features_dict)
    def transform_y(y):
        return np.eye(len(target_labels))[target_labels.index(y)]
    dataset.set_transform_y(transform_y)
    dataset.set_size_version_2(True)
    dataset.set_limit_batches(batches)
    dataset.shuffle_df()

    return dataset

class DatasetDictionaryManager:
    def __init__(self, train_dataset, val_dataset, test_dataset, target_labels):
        self.train_X = train_dataset.get_all_xs()
        self.train_Y = train_dataset.get_all_ys()
        self.train_class_weights = train_dataset.get_class_weights(target_labels)
        self.val_X = val_dataset.get_all_xs()
        self.val_Y = val_dataset.get_all_ys()
        self.test_X = test_dataset.get_all_xs()
        self.test_Y = test_dataset.get_all_ys()

    def get_dataset_dictionary(self, leads):
        return {
            'train': {
                'xs': self.train_X[:, leads, :],
                'ys': self.train_Y,
                'class_weights': self.train_class_weights
            },
            'val': {
                'xs': self.val_X[:, leads, :],
                'ys': self.val_Y
            },
            'test': {
                'xs': self.test_X[:, leads, :],
                'ys': self.test_Y
            }
        }

def get_shap_vals(model, x_train, x_test, y_test, x_train_samples, x_test_samples):
    x_train_indices = np.random.choice(len(x_train), min(x_train_samples, len(x_train)), replace=False)
    x_test_indices = np.random.choice(len(x_test), min(x_test_samples, len(x_test)), replace=False)
    
    shap_dp = shap.DeepExplainer(model, x_train[x_train_indices])
    shap_vals = shap_dp.shap_values(x_test[x_test_indices])
    
    return shap_vals, np.argmax(y_test, axis=1)

def percentages(arr):
    return arr / np.max(arr)

def custom_metric(y_true, y_pred):
    # Cast the predictions and true labels to binary values (0 or 1)
    y_pred = tf.round(y_pred)
    y_true = tf.cast(y_true, dtype=tf.float32)

    # Calculate true positives, false positives, and false negatives
    tp = tf.reduce_sum(y_true * y_pred, axis=0)
    fp = tf.reduce_sum(y_pred, axis=0) - tp
    fn = tf.reduce_sum(y_true, axis=0) - tp

    # Calculate precision and recall avoiding division by zero
    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    recall = tp / (tp + fn + tf.keras.backend.epsilon())

    # Calculate F1 score per class
    f1_per_class = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())

    # Calculate macro F1 score
    f1_macro = tf.reduce_mean(f1_per_class)

    return f1_macro

def train_classifier(config, dataset_dictionary, target_labels, hyperparams, 
                     feat_num, lead_num):
    
    cls_report = {
        'shaps': [],
        'val': {
            'prob': [],
            'pred': [],
            'target': []
        },
        'test': {
            'prob': [],
            'pred': [],
            'target': []
        }
    }

    end = config['repeat_times']
    i = 0
    
    f1_val_scores = []
    f1_test_scores = []

    while i <= end:
        model = get_feats_model(
            feat_num, lead_num, hyperparams['num_of_dense_layers'],
            hyperparams['add_pooling'],
            hyperparams['dropout_rate']
        )
        optimizer = Adam()
        model.compile(optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=4, verbose=1, 
            mode='max', restore_best_weights=True)
        adaptive_lr = ReduceLROnPlateau(monitor='val_loss',
                                   mode='min',
                                    factor=0.1,
                                    patience=3,
                                    min_lr=0.001 / 100)

        model.fit(x=dataset_dictionary['train']['xs'], 
                y=dataset_dictionary['train']['ys'], 
                epochs=config['epochs'], 
                validation_data=(dataset_dictionary['val']['xs'], dataset_dictionary['val']['ys']),
                callbacks=[early_stopping],
                class_weight=dataset_dictionary['train']['class_weights'])
        Y_val_pred_proba = model.predict(dataset_dictionary['val']['xs'])
        Y_val_pred = np.argmax(Y_val_pred_proba, axis=1)
        Y_val = dataset_dictionary['val']['ys']

        Y_test_pred_proba = model.predict(dataset_dictionary['test']['xs'])
        Y_test_pred = np.argmax(Y_test_pred_proba, axis=1)
        Y_test = dataset_dictionary['test']['ys']

        f1_val_scores.append(sklearn.metrics.f1_score(Y_val_pred, np.argmax(Y_val, axis=1), average='macro'))
        f1_test_scores.append(sklearn.metrics.f1_score(Y_test_pred, np.argmax(Y_test, axis=1), average='macro'))    

        shap_values = get_shap_vals(
            model, dataset_dictionary['train']['xs'], dataset_dictionary['val']['xs'], 
            Y_val, config['shap_instances']['train'], config['shap_instances']['val'])
        shap_results = np.array(shap_values[0])
        cls_report['shaps'].append(shap_results)
        cls_report['val']['prob'].append(Y_val_pred_proba)
        cls_report['val']['pred'].append(Y_val_pred)
        cls_report['val']['target'].append(np.argmax(Y_val, axis=1))
        cls_report['test']['prob'].append(Y_test_pred_proba)
        cls_report['test']['pred'].append(Y_test_pred)
        cls_report['test']['target'].append(np.argmax(Y_test, axis=1))
        i += 1

    return cls_report

def get_lead_importances(cls_report):
    shaps = np.array(cls_report['shaps'])
    shaps = np.abs(shaps)
    divider = shaps.shape[2]
    shaps = np.sum(shaps, axis=(0, 1, 2, 4))
    return shaps / divider

def f1_score_for_cls_dict(cls_report):
    f1_scores = []
    for i in range(len(cls_report['val']['target'])):
        Y_val = cls_report['val']['target'][i]
        Y_val_pred = cls_report['val']['pred'][i]
        f1_score = sklearn.metrics.f1_score(Y_val, Y_val_pred, average='macro')
        f1_scores.append(f1_score)
    return np.mean(f1_scores)


def feature_elim_shap(ds_manager, config, target_labels):
    all_lead_names = LEAD_NAMES.copy()
    hyperparameters = read_config(config['hyperparameters_path'])

    lead_feature_importances = []
    lead_steps = []
    f1_scores = []
    classifier_dicts = []
    leads = [i for i in range(12)]

    for lead_iter in tqdm(range(12), desc='Lead iterations', leave=False):
        lead_num = 12 - lead_iter
        dataset_dictionary = ds_manager.get_dataset_dictionary(leads)
        classifier_dict = train_classifier(
            config, dataset_dictionary, target_labels, 
            hyperparameters[str(lead_num - 1)], FEAT_NUM, lead_num
        )
        lead_importances = get_lead_importances(classifier_dict)
        lowest_feat_imp_idx = np.argmin(lead_importances)

        classifier_dicts.append(classifier_dict)
        f1_scores.append(f1_score_for_cls_dict(classifier_dict))
        lead_feature_importances.append(lead_importances)
        lead_steps.append(all_lead_names.copy())
        del leads[lowest_feat_imp_idx]
        del all_lead_names[lowest_feat_imp_idx]
    
    report = {
        'classifier_dicts': classifier_dicts,
        'f1_scores': f1_scores,
        'lead_feature_importances': lead_feature_importances,
        'lead_steps': lead_steps
    }
    return report

def main(config):
    log_dict = {}
    start_time = time()
    features = np.load(f'{config["features_path"]}/{config["arrhythmia_label"]}.npy', allow_pickle=True).item()

    log_dict['config'] = config

    arrhythmia_label = config['arrhythmia_label']
    target_labels = [arrhythmia_label, 'NSR', 'Other']

    train_df = get_dataset_df(config, config['arrhythmia_label'], config['description_file_name'])
    val_df = get_dataset_df(config, config['arrhythmia_label'], config['val_description_file_name'])
    test_df = get_dataset_df(config, config['arrhythmia_label'], config['test_description_file_name'])

    train_dataset = get_dataset(train_df, config, config["batches"]["train"], config['feats_num'], target_labels, features)
    val_dataset = get_dataset(val_df, config, config["batches"]["val"], config['feats_num'], target_labels, features)
    test_dataset = get_dataset(test_df, config, config["batches"]["test"], config['feats_num'], target_labels, features)

    train_dataset.balance_by_max_batch_size()

    ds_manager = DatasetDictionaryManager(train_dataset, val_dataset, test_dataset, target_labels)

    report = feature_elim_shap(ds_manager, config, target_labels)

    end_time = time()
    log_dict['report'] = report
    log_dict['duration'] = end_time - start_time
    save_log_as_npy(log_dict, config['log_path'])

if __name__ == "__main__":
    config = read_config(f"{DIR_PATH}/configs/dl_optimal_subsets.json")
    main(config)
    