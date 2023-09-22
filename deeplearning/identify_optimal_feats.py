import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
for _ in range(2):
    current = os.path.dirname(current)
sys.path.append(current)

from utils.constants import ARRHYTHMIA_LABELS
from utils.functions import read_config, get_dataset_df, get_all_records
from models import get_extr_feats_model_multiple_feats_extr, optimal_feats_model
import pandas as pd
from datahelpers.dldata import DLDatasetNoBatch, DLDataset
from keras.models import Model
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import sklearn.metrics
import json
from dl_utils import get_whole_dataset

DIR_PATH = "deeplearning/with_feats_extr"

def get_features_dict(model, data, record_names):
    predicted_features = model.predict(data)

    features_dict = {}

    for i in tqdm(range(len(record_names)), desc='get_features_dict', leave=False):
        features_dict[record_names[i]] = np.expand_dims(predicted_features[i], axis=1)
    
    return features_dict

def get_dataset(dataframe, config, sequence_len, target_labels, features_dict, lead):
    dataset = DLDataset(
        dataframe, config['dataset_path'], record_len=sequence_len)
    dataset.set_leads([0])
    dataset.set_record_lookup(features_dict)
    def transform_y(y):
        return np.eye(len(target_labels))[target_labels.index(y)]
    dataset.set_transform_y(transform_y)
    return dataset


def train_and_get_f1_score(config, model, train_df, val_df, arrhythmia_label, data, record_names, feat_num, lead):
    features_dict = get_features_dict(model, data, record_names)
    target_labels = [arrhythmia_label, 'NSR', 'Other']

    train_dataset = get_dataset(train_df, config, feat_num, target_labels, features_dict, lead)
    val_dataset = get_dataset(val_df, config, feat_num, target_labels, features_dict, lead)
    train_dataset.shuffle_df()

    new_model = optimal_feats_model(feat_num)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=7, verbose=1, 
        mode='min', restore_best_weights=True)

    new_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    new_model.fit(train_dataset, epochs=10, validation_data=val_dataset, callbacks=[early_stopping])
    val_predictions = new_model.predict(val_dataset)
    val_true = val_dataset.get_all_ys()
    return sklearn.metrics.f1_score(np.argmax(val_true, axis=1), 
                                    np.argmax(val_predictions, axis=1), average='macro')

def get_results_arrhhythm_lead(config, arrhythmia_label, lead):
    model = get_extr_feats_model_multiple_feats_extr(3, config['record_len'])
    train_df = get_dataset_df(config, arrhythmia_label, config['description_file_name'])
    val_df = get_dataset_df(config, arrhythmia_label, config['val_description_file_name'])

    model.load_weights(f'{config["models_path"]}/{arrhythmia_label}_{lead}.h5')
    data, record_names = get_whole_dataset(config, lead)

    models_feats = {
        4: Model(model.input, model.layers[-2].output),
        8: Model(model.input, model.layers[-3].output),
        16: Model(model.input, model.layers[-4].output),
        32: Model(model.input, model.layers[-5].output),
        64: Model(model.input, model.layers[-6].output),
    }

    results = {}

    for feat_num in [4, 8, 16, 32, 64]:
        results[feat_num] = train_and_get_f1_score(config, models_feats[feat_num], train_df, 
                                val_df, arrhythmia_label, 
                                data, record_names, feat_num, lead)
    return results

def main(config):
    results = {}
    for tuple in config["combinations"]:
        arrhythmia_label = tuple[0]
        lead = tuple[1]
        res = []
        res.append(get_results_arrhhythm_lead(config, arrhythmia_label, lead))
        results[f"{arrhythmia_label}_{lead}"] = res
    json.dump(results, open(f'{config["save_log_path"]}/optimal_feats_16-64.json', 'w'))

if __name__ == "__main__":
    config = read_config(f'{DIR_PATH}/configs/identify_optimal_feats.json')
    main(config)