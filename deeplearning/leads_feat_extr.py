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

DIR_PATH = "deeplearning/with_feats_extr"

# Optimization settings
LOSS_NAME = 'binary_crossentropy'

def get_dataset(dataframe, config, sequence_len, target_labels):
    dataset = dldata.DLDataset(dataframe, config['dataset_path'], record_len=sequence_len)
    def transform_y(y):
        return np.eye(len(target_labels))[target_labels.index(y)]
    dataset.set_transform_y(transform_y)
    return dataset

# Save final result
def get_pred(dataset, model):
    y_score = model.predict(dataset, verbose=1)

    json_pred = {
        "pred": y_score,
        "true": dataset.get_all_ys()
    }

    return json_pred

def get_datasets(config):
    train_df = get_dataset_df(config, config['arrhythmia_label'], config['description_file_name'])
    val_df = get_dataset_df(config, config['arrhythmia_label'], config['val_description_file_name'])
    test_df = get_dataset_df(config, config['arrhythmia_label'], config['test_description_file_name'])

    target_labels = [config['arrhythmia_label'], 'NSR', 'Other']
    
    train_dataset = get_dataset(train_df, config, config['record_len'], target_labels)
    val_dataset = get_dataset(val_df, config, config['record_len'], target_labels)
    test_dataset = get_dataset(test_df, config, config['record_len'], target_labels)

    train_dataset.set_limit_batches(config['train_batches'])
    train_dataset.balance_by_max_batch_size()
    val_dataset.set_limit_batches(config['val_batches'])
    test_dataset.set_limit_batches(config['test_batches'])
    for dataset in [train_dataset, val_dataset, test_dataset]:
        dataset.shuffle_df()
        dataset.set_leads([config['lead']])

    return train_dataset, val_dataset, test_dataset

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
    data = np.array([dl_dataset[i][0] for i in range(len(dl_dataset))])
    return data, records_names

def main(config):
    train_dataset, val_dataset, _ = get_datasets(config)

    opt = Adam(config['learning_rate'])
    callbacks = [ReduceLROnPlateau(monitor='val_loss',
                                   mode='min',
                                    factor=0.1,
                                    patience=3,
                                    min_lr=config['learning_rate'] / 100),
                    EarlyStopping(monitor='val_loss',
                                patience=5,  # Patience should be larger than the one in ReduceLROnPlateau
                                min_delta=0.00001,
                                restore_best_weights=True)]

    model = get_extr_feats_model_multiple_feats_extr(3, config['record_len'])
    model.compile(loss=LOSS_NAME, optimizer=opt, metrics=['acc'])

    train_y = train_dataset.get_all_ys()
    lbl_train_y = np.argmax(train_y, axis=1)
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced', classes=np.unique(lbl_train_y), y=lbl_train_y)
    class_weights = { i : class_weights[i] for i in range(len(class_weights)) }
    history = model.fit(train_dataset,
                        epochs=config['epochs'],
                        initial_epoch=0,  # If you are continuing a interrupted section change here
                        callbacks=callbacks,
                        validation_data=val_dataset,
                        verbose=1,
                        class_weight=class_weights)
    model.save_weights(f"{config['save_models_path']}/{config['arrhythmia_label']}_{config['lead']}.h5")

if __name__ == "__main__":
    config = read_config(f'{DIR_PATH}/configs/leads_feat_extr.json')
    main(config)