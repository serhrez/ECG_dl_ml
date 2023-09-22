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
from utils.functions import read_config, get_dataset_df, get_all_records
import datahelpers.dldata as dldata
import math
from keras.utils import Sequence
from models import get_extr_feats_model_multiple_feats_extr, model_with_needed_layers, get_feats_model
import pandas as pd
from datahelpers.dldata import DLDatasetNoBatch, DLDataset
import tensorflow as tf
from tqdm import tqdm
import random
import time
from kerastuner import RandomSearch
import json

DIR_PATH = "deeplearning/with_feats_extr"

def get_whole_dataset(config):
    records_names = get_all_records(config, config['whole_description_file_name'])
    records_names = records_names
    whole_df = pd.DataFrame({
        'Record': records_names,
        'Arrhythmia': ['Dummy'] * len(records_names)
    })
    dl_dataset = DLDatasetNoBatch(whole_df, config['dataset_path'], config['record_len'])
    dl_dataset.set_transform_y(lambda _: [0, 0, 0])
    dl_dataset.set_leads([config['lead']])
    data = np.array([dl_dataset[i][0] for i in tqdm(range(len(dl_dataset)), desc='get_whole_dataset', leave=False)])
    return data, records_names

def get_dataset(dataframe, config, sequence_len, target_labels, features_dict):
    dataset = dldata.DLDataset(
        dataframe, config['dataset_path'], record_len=sequence_len)
    dataset.set_record_lookup(features_dict)
    def transform_y(y):
        return np.eye(len(target_labels))[target_labels.index(y)]
    dataset.set_transform_y(transform_y)
    dataset.set_size_version_2(True)
    return dataset

def get_features_dict(model, data, record_names):
    predicted_features = model.predict(data)

    features_dict = {}

    for i in range(len(record_names)):
        features_dict[record_names[i]] = np.expand_dims(predicted_features[i], axis=1)
    
    return features_dict

import tensorflow as tf

def get_model_function(params_dict):
    def get_model(hp):
        num_of_dense_layers = hp.Choice('num_of_dense_layers', values=params_dict['num_of_dense_layers'])
        add_pooling = hp.Choice('add_pooling', values=params_dict['add_pooling'])
        dropout_rate = hp.Choice('dropout_rate', values=params_dict['dropout_rate'])    
        lr = hp.Choice('learning_rate', values=params_dict['learning_rate'])
        model = get_feats_model(
            32, params_dict['number_of_leads'], num_of_dense_layers, 
            add_pooling, dropout_rate)
        optimizer = Adam(learning_rate=lr)

        model.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
        return model
    return get_model
    
def get_params_dict(num_of_leads):
    params_dict = {}
    params_dict['num_of_dense_layers'] = [2, 3, 4, 5, 6]
    params_dict['add_pooling'] = [True, False]
    params_dict['dropout_rate'] = [0.0, 0.1, 0.2, 0.3]
    params_dict['learning_rate'] = [0.01, 0.001, 0.0001]
    params_dict['number_of_leads'] = num_of_leads
    return params_dict


def main(config):
    features = np.load(f'{config["features_path"]}/{config["arrhythmia_label"]}.npy', allow_pickle=True).item()
    feat_num = 32
    leads = list(range(12))[:config['number_of_leads']]

    train_df = get_dataset_df(config, config['arrhythmia_label'], config['description_file_name'])
    val_df = get_dataset_df(config, config['arrhythmia_label'], config['val_description_file_name'])

    target_labels = [config['arrhythmia_label'], 'NSR', 'Other']

    train_dataset = get_dataset(train_df, config, feat_num, target_labels, features)
    val_dataset = get_dataset(val_df, config, feat_num, target_labels, features)
    train_dataset.set_leads(leads)
    val_dataset.set_leads(leads)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=2, verbose=1, 
        mode='min', restore_best_weights=True)
    tuner = RandomSearch(
        get_model_function(get_params_dict(config['number_of_leads'])),
        objective='val_accuracy',
        max_trials=40,  # Number of different hyperparameter combinations to try,
        directory=config['directory'],
        project_name=f'{config["arrhythmia_label"]}_{config["number_of_leads"]}',
        )

    tuner.search(train_dataset, epochs=3, validation_data=val_dataset, callbacks=[early_stopping], verbose=1)

    best_values = 5

    best_hyperparams = [trial.hyperparameters.values for trial in tuner.oracle.get_best_trials(best_values)]
    best_accuracies = [trial.metrics.metrics['val_accuracy'].get_statistics()['mean'] for trial in tuner.oracle.get_best_trials(best_values)]
    
    hyperparam_dict = {
        "best_hyperparams": best_hyperparams,
        "best_accuracies": best_accuracies
    }
    with open(f'{config["hyperparams_logs"]}/{config["arrhythmia_label"]}_{config["number_of_leads"]}.json', 'w') as f:
        json.dump(hyperparam_dict, f, indent=2)


    pass


    
    
if __name__ == "__main__":
    config = read_config(f'{DIR_PATH}/configs/optimize_hyperparams.json')
    main(config)