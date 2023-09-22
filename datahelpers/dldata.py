import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
for _ in range(1):
    current = os.path.dirname(current)
sys.path.append(current)

import scipy.io as sio
import os
import numpy as np
import keras
import tensorflow as tf
import pandas as pd
from torch.utils.data import Dataset
from utils.functions import resample_to_500hz

def do_nothing(x):
    return x

class DLDataset(keras.utils.Sequence, Dataset):
    def __init__(self, dataframe, dataset_path, record_len, batch_size=32):
        self.dataframe = dataframe
        self.dataset_path = dataset_path
        self.record_len = record_len
        self.batch_size = batch_size
        self.transform_y = do_nothing
        self.transform_x = do_nothing
        self.limit_batches = None
        self.leads = [i for i in range(12)]
        self.record_lookup = None
        self.size_version_2 = False
        self.custom_make_record = None
        self.should_resample = False

    def set_custom_make_record(self, custom_make_record):
        self.custom_make_record = custom_make_record

    def set_size_version_2(self, val):
        self.size_version_2 = val

    def set_record_lookup(self, record_lookup):
        self.record_lookup = record_lookup

    def set_transform_y(self, transform_y):
        self.transform_y = transform_y

    def set_transform_x(self, transform_x):
        self.transform_x = transform_x

    def set_limit_batches(self, limit_batches):
        self.limit_batches = limit_batches

    def get_class_weights(self, target_labels):
        arrhythmias = self.dataframe['Arrhythmia'].iloc[:len(self) * self.batch_size]
        unique, counts = np.unique(arrhythmias, return_counts=True)
        class_weights = dict(zip(unique, counts))
        class_weights = {target_labels.index(key): max(counts) / value for key, value in class_weights.items()}
        return class_weights
    
    def balance_samples_in_classes(self):
        arrhythmias_pds = []
        for arrhythm in self.dataframe['Arrhythmia'].unique():
            arrhythmias_pds.append(self.dataframe[self.dataframe['Arrhythmia'] == arrhythm])
        min_len = min([len(arrhythmia_pd) for arrhythmia_pd in arrhythmias_pds])
        self.dataframe = pd.concat([arrhythmia_pd.sample(min_len) for arrhythmia_pd in arrhythmias_pds])

    def set_leads(self, leads):
        self.leads = leads

    def balance_by_max_batch_size(self):
        max_length = self.limit_batches * self.batch_size
        arrhythmias_pds = []
        for arrhythm in self.dataframe['Arrhythmia'].unique():
            arrhythmias_pds.append(self.dataframe[self.dataframe['Arrhythmia'] == arrhythm])
        allocated_lengths = [max_length // len(arrhythmias_pds)] * len(arrhythmias_pds)
        # Sort allocated lengths indices
        allocated_lengths_indices = np.argsort([len(arr) for arr in arrhythmias_pds])
        arrhythmias_pds = [arrhythmias_pds[i] for i in allocated_lengths_indices]
        allocated_lengths = [allocated_lengths[i] for i in allocated_lengths_indices]
        for i in range(len(allocated_lengths) - 1):
            if len(arrhythmias_pds[i]) < allocated_lengths[i]:
                new_length = len(arrhythmias_pds[i])
                remaining = (allocated_lengths[i] - new_length) // (len(allocated_lengths) - i - 1)
                for j in range(i + 1, len(allocated_lengths)):
                    allocated_lengths[j] = allocated_lengths[j] + remaining
                allocated_lengths[i] = new_length
        
        self.dataframe = pd.concat([arrhythmia_pd.sample(min(allocated_lengths[i], len(arrhythmia_pd))) for i, arrhythmia_pd in enumerate(arrhythmias_pds)])

    def shuffle_df(self):
        self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)

    def __len__(self):
        num_batches = int(np.floor(len(self.dataframe) / self.batch_size))
        return min(num_batches, self.limit_batches or num_batches)
    
    def __getitem__(self, idx):
        df_range = self.dataframe.iloc[idx * self.batch_size : (idx + 1) * self.batch_size]
        arrhythmias = df_range['Arrhythmia'].iloc
        # Read numpy array from .mat file
        record_names = df_range['Record'].iloc
        if self.size_version_2:
            records_output = np.empty((self.batch_size, len(self.leads), self.record_len))
        else:
            records_output = np.empty((self.batch_size, self.record_len, len(self.leads)))
        arrhythmias_output = np.empty((self.batch_size, 3))

        for i in range(self.batch_size):
            records_output[i] = self._get_record(record_names[i])
            arrhythmias_output[i] = self.transform_y(arrhythmias[i])
        return records_output, arrhythmias_output
    
    def _get_record(self, record_name):
        if self.record_lookup == None:
            record_path = os.path.join(self.dataset_path, record_name)
            record = sio.loadmat(record_path)['val']
            with open(record_path + '.hea', 'r') as f:
                lines = f.readlines()
                frequency = int(lines[0].split(' ')[2])
            if self.should_resample:
                record = resample_to_500hz(record, frequency)
            record = self._make_record(record)
            return self.transform_x(record[:, self.leads])
        else:
            if self.size_version_2:
                rec = self.record_lookup[record_name][self.leads, :]
            else:
                rec = self.record_lookup[record_name][:, self.leads]
            return self.transform_x(rec)
        
    def _make_record(self, record):
        if self.custom_make_record != None:
            record = self.custom_make_record(record)
        if record.shape[1] < self.record_len:
            record = np.concatenate((record, np.zeros((record.shape[0], self.record_len - record.shape[1]))), axis=1)
        elif record.shape[1] > self.record_len:
            record = record[:, :self.record_len]
        record = record.reshape((record.shape[1], record.shape[0]))
        return record
    
    def get_all_ys(self):
        y = self.dataframe['Arrhythmia'].iloc[:len(self) * self.batch_size].values
        y_output = np.empty((len(y), 3))
        for i in range(len(y)):
            y_output[i] = self.transform_y(y[i])
        return y_output
    
    def get_all_xs(self):
        batches = [self[idx] for idx in range(len(self))]
        if self.size_version_2:
            x = np.empty((len(batches) * self.batch_size, len(self.leads), self.record_len))
        else:
            x = np.empty((len(batches) * self.batch_size, self.record_len, len(self.leads)))
        for i, batch in enumerate(batches):
            x[i * self.batch_size : (i + 1) * self.batch_size] = batch[0]
        return x

class DLDatasetNoBatch(keras.utils.Sequence, Dataset):
    def __init__(self, dataframe, dataset_path, record_len):
        self.dataframe = dataframe
        self.dataset_path = dataset_path
        self.record_len = record_len
        self.transform_y = do_nothing
        self.transform_x = do_nothing
        self.leads = [i for i in range(12)]
        self.custom_make_record = None
        self.should_resample = False

    def set_leads(self, leads):
        self.leads = leads

    def set_transform_y(self, transform_y):
        self.transform_y = transform_y

    def set_transform_x(self, transform_x):
        self.transform_x = transform_x

    def set_custom_make_record(self, custom_make_record):
        self.custom_make_record = custom_make_record

    def get_class_weights(self, target_labels):
        arrhythmias = self.dataframe['Arrhythmia'].iloc[:len(self) * self.batch_size]
        unique, counts = np.unique(arrhythmias, return_counts=True)
        class_weights = dict(zip(unique, counts))
        class_weights = {target_labels.index(key): max(counts) / value for key, value in class_weights.items()}
        return class_weights
    
    def balance_samples_in_classes(self):
        arrhythmias_pds = []
        for arrhythm in self.dataframe['Arrhythmia'].unique():
            arrhythmias_pds.append(self.dataframe[self.dataframe['Arrhythmia'] == arrhythm])
        min_len = min([len(arrhythmia_pd) for arrhythmia_pd in arrhythmias_pds])
        self.dataframe = pd.concat([arrhythmia_pd.sample(min_len) for arrhythmia_pd in arrhythmias_pds])
        # Shuffle rows in dataframe
        self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        df = self.dataframe.iloc[idx]
        arrhythmia = df['Arrhythmia']
        # Read numpy array from .mat file
        record_name = df['Record']
        arrhythmias_output = np.empty((3))
        record_path = os.path.join(self.dataset_path, record_name)
        record = sio.loadmat(record_path)['val']
        with open(record_path + '.hea', 'r') as f:
            lines = f.readlines()
            frequency = int(lines[0].split(' ')[2])
        if self.should_resample:
            record = resample_to_500hz(record, frequency)
        record = self._make_record(record)
        records_output = self.transform_x(record[:, self.leads])
        arrhythmias_output = self.transform_y(arrhythmia)
        return records_output, arrhythmias_output
    
    def _make_record(self, record):            
        if self.custom_make_record != None:
            record = self.custom_make_record(record)
        if record.shape[1] < self.record_len:
            record = np.concatenate((record, np.zeros((record.shape[0], self.record_len - record.shape[1]))), axis=1)
        elif record.shape[1] > self.record_len:
            record = record[:, :self.record_len]
        record = record.reshape((record.shape[1], record.shape[0]))
        return record

class DLDatasetSimple(keras.utils.Sequence):
    def __init__(self, dataframe, dataset_path, record_len):
        self.dataframe = dataframe
        self.dataset_path = dataset_path
        self.record_len = record_len
        self.transform_y = do_nothing
        self.transform_x = do_nothing

    def set_transform_y(self, transform_y):
        self.transform_y = transform_y

    def set_transform_x(self, transform_x):
        self.transform_x = transform_x