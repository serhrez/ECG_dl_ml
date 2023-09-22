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
from models import get_extr_feats_model_multiple_feats_extr, model_with_needed_layers
import pandas as pd
from datahelpers.dldata import DLDatasetNoBatch, DLDataset
import tensorflow as tf
from tqdm import tqdm

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
    dataset.set_leads([0])
    dataset.set_record_lookup(features_dict)
    def transform_y(y):
        return np.eye(len(target_labels))[target_labels.index(y)]
    dataset.set_transform_y(transform_y)
    return dataset

def get_features_dict(model, data, record_names):
    predicted_features = model.predict(data)

    features_dict = {}

    for i in range(len(record_names)):
        features_dict[record_names[i]] = np.expand_dims(predicted_features[i], axis=1)
    
    return features_dict

import tensorflow as tf

def create_feedforward_nn(input_size):
    # Define the model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(input_size,1)),
        tf.keras.layers.Dense(64, activation='relu'),  # Hidden layer with 64 neurons and ReLU activation
        tf.keras.layers.Dense(3, activation='softmax')  # Output layer with 3 neurons and softmax activation for multi-class classification
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

DIR_PATH = "deeplearning/with_feats_extr"
config = read_config(f'{DIR_PATH}/configs/leads_feat_extr.json')
config['lead'] = 8
LOSS_NAME = 'binary_crossentropy'
opt = Adam(lr=config['learning_rate'])

model = get_extr_feats_model_multiple_feats_extr(3, config['record_len'])
model.compile(loss=LOSS_NAME, optimizer=opt, metrics=['acc'])
model.load_weights(f'{DIR_PATH}/features_models/AF_0.h5')

data, record_names = get_whole_dataset(config)
models_feats = {
    4: Model(model.input, model.layers[-2].output),
    8: Model(model.input, model.layers[-3].output),
    16: Model(model.input, model.layers[-4].output),
    32: Model(model.input, model.layers[-5].output),
    64: Model(model.input, model.layers[-6].output)
}

def print_results(feat_num):
    features_dict = get_features_dict(models_feats[feat_num], data, record_names)

    train_df = get_dataset_df(config, config['arrhythmia_label'], config['description_file_name'])
    val_df = get_dataset_df(config, config['arrhythmia_label'], config['val_description_file_name'])
    test_df = get_dataset_df(config, config['arrhythmia_label'], config['test_description_file_name'])

    target_labels = [config['arrhythmia_label'], 'NSR', 'Other']

    train_dataset = get_dataset(train_df, config, feat_num, target_labels, features_dict)
    val_dataset = get_dataset(val_df, config, feat_num, target_labels, features_dict)
    test_dataset = get_dataset(test_df, config, feat_num, target_labels, features_dict)

    model = create_feedforward_nn(feat_num)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=7, verbose=1, 
        mode='min', restore_best_weights=True)

    model.fit(train_dataset, epochs=20, validation_data=val_dataset, callbacks=[early_stopping])
    print(f"Number of feats: {feat_num}")
    print("Train classification report:")
    print(sklearn.metrics.classification_report(
        np.argmax(train_dataset.get_all_ys(), axis=1), 
        np.argmax(model.predict(train_dataset), axis=1), 
        target_names=target_labels))
    print("Val classification report:")
    print(sklearn.metrics.classification_report(
        np.argmax(val_dataset.get_all_ys(), axis=1),
        np.argmax(model.predict(val_dataset), axis=1),
        target_names=target_labels))
    print("Test classification report:")
    print(sklearn.metrics.classification_report(
        np.argmax(test_dataset.get_all_ys(), axis=1),
        np.argmax(model.predict(test_dataset), axis=1),
        target_names=target_labels))
    pass

print_results(32)
