from dataset import Dataset
from datahelpers import dldata
import torch
import torch.nn as nn
import numpy as np
import random
import pandas as pd

import torch.nn as nn
from keras.layers import (
    Input, Conv1D, MaxPooling1D, Dropout, BatchNormalization, Activation, Add, Flatten, Dense)
from keras.models import Model
from utils.functions import get_dataset_dataframe
import sklearn.metrics
from sklearn.model_selection import train_test_split
from utils.functions import identify_optimal_thresholds

def get_model(n_classes, length, input_classes, last_layer='sigmoid'):
    kernel_size = 16
    kernel_initializer = 'he_normal'
    signal = Input(shape=(length, input_classes), dtype=np.float32, name='signal')
    x = signal
    num_filters = 128
    for i in range(8):
        x = Conv1D(num_filters, kernel_size, padding='same', use_bias=False,
                kernel_initializer=kernel_initializer)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        if i % 2 == 1:
            num_filters /= 2
        if i < 4:
            x = MaxPooling1D(4)(x)

    x = Flatten()(x)
    diagn = Dense(n_classes, activation=last_layer, kernel_initializer=kernel_initializer)(x)
    model = Model(signal, diagn)
    return model


config = {
    "arrhythmia_label": "LBBB",
    "dataset_path": "D:\\research_old\\research_large_files\\card_challenge\\training",
    "description_file_name": "train_description.json",
    "features_path": "features/all/whole_ds_processed.csv",
    "ds_class_module": "datahelpers.allpreddata",
    "ds_class_name": "AllPredData",
    "log_name": "rf_backward_stepwise",
    "n_samples": 5000,
    "highly_corr_feats_threshold": 0.3,
    "test_description_file_name": "test_description.json"
}
target_labels = ['LBBB', 'NSR', 'Other']

def get_dataset(dataframe):
    dataset = dldata.DLDataset(dataframe, config['dataset_path'], record_len=5000)
    def transform_y(y):
        return np.eye(len(target_labels))[target_labels.index(y)]
    dataset.set_transform_y(transform_y)
    return dataset

train_val_df = get_dataset_dataframe(config, config['arrhythmia_label'], False)
train_df, val_df = train_test_split(train_val_df, test_size=0.15)
test_df = get_dataset_dataframe(config, config['arrhythmia_label'], True)

train_dataset = get_dataset(train_df)
val_dataset = get_dataset(val_df)
test_dataset = get_dataset(test_df)
train_dataset.set_limit_batches(200)
val_dataset.set_limit_batches(100)
test_dataset.set_limit_batches(100)

train_dataset.shuffle_df()
val_dataset.shuffle_df()
test_dataset.shuffle_df()



model = get_model(3, 5000, 12)
import tensorflow_addons as tfa

model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy', tfa.metrics.F1Score(num_classes=3, average='macro')])
 
# Save the best model by validation accuracy
checkpoint_filepath = 'best_model.h5'
import keras
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor=tfa.metrics.F1Score(num_classes=3, average='macro'),
    mode='max',
    save_best_only=True)

class_weight = train_dataset.get_class_weights()
class_weight = {target_labels.index(key) : value for key, value in class_weight.items()}
model.fit(train_dataset, epochs=10, validation_data=val_dataset, 
          verbose=1, callbacks=[model_checkpoint_callback])
model.load_weights(checkpoint_filepath)

# Evaluate on train set
preds = model.predict(train_dataset)
preds = np.argmax(preds, axis=1)
train_Y = np.argmax(train_dataset.get_all_ys(), axis=1)
cl_report = sklearn.metrics.classification_report(train_Y, preds, target_names=target_labels)
print("Accuracy train: ", cl_report)

# Evaluate on val set
preds = model.predict(val_dataset)
optimal_thresholds, _ = identify_optimal_thresholds(val_dataset.get_all_ys(), preds)
preds = (preds >= optimal_thresholds).astype(int)
preds = np.argmax(preds, axis=1)
val_Y = np.argmax(val_dataset.get_all_ys(), axis=1)
cl_report = sklearn.metrics.classification_report(val_Y, preds, target_names=target_labels)
print("Accuracy val: ", cl_report)

# Evaluate on test set
preds = model.predict(test_dataset)
preds = (preds >= optimal_thresholds).astype(int)
preds = np.argmax(preds, axis=1)
test_Y = np.argmax(test_dataset.get_all_ys(), axis=1)
cl_report = sklearn.metrics.classification_report(test_Y, preds, target_names=target_labels)
print("Accuracy test: ", cl_report)


pass