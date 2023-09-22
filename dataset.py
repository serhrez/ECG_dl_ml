# Add the directory of the parent folder to the system path
import sys
sys.path.append('..')

import json
import os
import pandas as pd
from utils.snomed_ct import get_snomed, get_arrhythmia
from sklearn.model_selection import train_test_split
import random

class RecordDescription:
    def __init__(self, dataset, arrhythmias):
        self.dataset = dataset
        self.arrhythmias = arrhythmias

    def is_multiple(self):
        return len(self.arrhythmias) > 1

class Dataset:    
    def __init__(self, dataset_dir, description_file_name):
        self.dataset_dir = dataset_dir
        self.description_file_name = description_file_name
        self.__init_description__()
        self.__init_records__()
            
    def __init_description__(self):
        path = os.path.join(self.dataset_dir, self.description_file_name)
        with open(path, 'r') as f:
            self._description = json.load(f)
        
    def __init_records__(self):
        self._records_descriptions = {}
        self._records_by_arrhythmia = {}
        for dataset in self._description:
            ds = self._description[dataset]
            for arrhythmia in ds:
                records_multi = ds[arrhythmia]
                for record in records_multi:
                    if record in self._records_descriptions:
                        self._records_descriptions[record].arrhythmias.append(arrhythmia)
                    else:
                        self._records_descriptions[record] = RecordDescription(dataset, [arrhythmia])

                    if arrhythmia in self._records_by_arrhythmia:
                        self._records_by_arrhythmia[arrhythmia].append(record)
                    else:
                        self._records_by_arrhythmia[arrhythmia] = [record]

    def get_all_records(self):
        return list(self._records_descriptions.keys())

    def get_pandas_dataframe(self, arrhythmia_label):
        snomed_arrhythmia = get_snomed(arrhythmia_label)
        arrhythmia_records = self._records_by_arrhythmia[snomed_arrhythmia]
        nsr_label = 'NSR'
        nsr_snomed = get_snomed(nsr_label)
        nsr_records = self._records_by_arrhythmia[nsr_snomed]
        nsr_records_filtered = []
        for record in nsr_records:
            if len(self._records_descriptions[record].arrhythmias) == 1:
                nsr_records_filtered.append(record)
        
        other_records = []
        for record in self._records_descriptions:
            descr = self._records_descriptions[record]
            if snomed_arrhythmia not in descr.arrhythmias and nsr_snomed not in descr.arrhythmias:
                other_records.append(record)
        

        record_names = arrhythmia_records + nsr_records_filtered + other_records
        arrhythmias = [arrhythmia_label] * len(arrhythmia_records) + \
                      [nsr_label] * len(nsr_records_filtered) + \
                      ['Other'] * len(other_records)
        is_multiple_arrhythmias = [self._records_descriptions[record].is_multiple() for record in record_names]
        datasets = [self._records_descriptions[record].dataset for record in record_names]
        df = pd.DataFrame({
            'Record': record_names, 
            'Arrhythmia': arrhythmias, 
            'IsMultipleArrhythmias': is_multiple_arrhythmias, 
            'Dataset': datasets})

        return df
    
def split_dataset(df, n_splits):
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    return mskf.split(df['Arrhythmia'], df[['Record', 'IsMultipleArrhythmias', 'Dataset']])
   
def check_n_samples_labels(pandas_dataset, arrhythmia_label):
    # Get all the samples with arrhythmia 'AF' in column 'Arrhythmia'
    arr_specific = len(pandas_dataset[pandas_dataset['Arrhythmia'] == arrhythmia_label])
    arr_normal = len(pandas_dataset[pandas_dataset['Arrhythmia'] == 'NSR'])
    arr_other = len(pandas_dataset[pandas_dataset['Arrhythmia'] == 'Other'])
    not_multiple_arr = len(pandas_dataset[pandas_dataset['IsMultipleArrhythmias'] == False])
    multiple_arr = len(pandas_dataset[pandas_dataset['IsMultipleArrhythmias'] == True])
    total = len(pandas_dataset)
    
    datasets = pandas_dataset['Dataset'].unique()
    datasets_multiples = []
    for dataset in datasets:
        datasets_multiples.append(len(pandas_dataset[pandas_dataset['Dataset'] == dataset]))

    print("Arrhythmia: {}".format(arrhythmia_label))
    print("Specific arrhythmia: {} {:.2f}".format(arr_specific, arr_specific/total))
    print("Normal arrhythmia: {} {:.2f}".format(arr_normal, arr_normal/total))
    print("Other arrhythmia: {} {:.2f}".format(arr_other, arr_other/total))
    print("Not multiple arrhythmias: {} {:.2f}".format(not_multiple_arr, not_multiple_arr/total))
    print("Multiple arrhythmias: {} {:.2f}".format(multiple_arr, multiple_arr/total))
    for i, ds in enumerate(datasets):
        print("Dataset {}: {} {:.2f}".format(ds, datasets_multiples[i], datasets_multiples[i]/total))

    print("Total: {}".format(len(pandas_dataset)))
    print("Arrhythmia END")

def visualize_indices(trains, tests):
    import matplotlib.pyplot as plt
    for i, (train, test) in enumerate(zip(trains, tests)):
        plt.hlines(train, xmin=0 + i * 0.2, xmax=0.1 + i * 0.2, color='b', linewidth=1, label='Train')
        plt.hlines(test, xmin=0.1 + i * 0.2, xmax=0.2 + i * 0.2, color='r', linewidth=1, label='Test')

    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Splitting Visualization')

    # Display a legend
    plt.legend()

    # Show the plot
    plt.show()

if __name__ == '__main__':
    from utils.functions import read_config
    config = read_config('baseline_predictor/config.json')
    dataset = Dataset(config['dataset_path'], 
        config['description_train'])
    
    arrhythmia_label = 'AF'
    df = dataset.get_pandas_dataframe(arrhythmia_label)
    print("=" * 20)
    print("ORIGINAL DATASET")
    check_n_samples_labels(df, arrhythmia_label)
    train_indices, test_indices = split_dataset(df, 5)
    trains = []
    tests = []
    for train_index, test_index in zip(train_indices, test_indices):
        trains.append(train_index)
        tests.append(test_index)
        df_train = df.iloc[train_index]
        df_test = df.iloc[test_index]
        print("=" * 20)
        print("TRAIN DATASET")
        check_n_samples_labels(df_train, arrhythmia_label)
        print("=" * 20)
        print("TEST DATASET")
        check_n_samples_labels(df_test, arrhythmia_label)
        x__ = 0
    visualize_indices(trains, tests)
    print(dataset.length())