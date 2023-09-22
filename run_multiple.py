from optimal_subsets import main as optimal_subsets_main
from extr_results import main as extr_results_main
from optimize_hyperparam import main as optimize_hyperparam_main
from utils.functions import read_config
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import utils.constants as constants
import numpy as np

def run_multiple_optimal_subsets():
    hyperparams_config = read_config('configs/hyperparams.json')
    config = read_config('configs/optimal_subsets.json')
    arrhythmias = ['AF', 'LBBB', 'RBBB', 'IAVB', 'PAC', 'PVC', 'STD', 'STE']
    for arrhythmia in tqdm(arrhythmias, desc='Arrhythmias'):
        config['arrhythmia_label'] = arrhythmia
        config['log_name'] = f'optimal_subsets/rf/os_rec_rf_{arrhythmia}'
        config['classifier_parameters'] = hyperparams_config[arrhythmia]
        config['optimal_thresholds'] = hyperparams_config[f'{arrhythmia}_optimal_thresholds']
        optimal_subsets_main(config)

def run_multiple_extr_results(version):
    if version == 'dl':
        save_directory = 'dl'
        log_directory = 'dl_converted'
        should_identify_thresholds = False
    elif version == 'rf':
        save_directory = 'rf'
        log_directory = 'rf'
        should_identify_thresholds = True
    config = read_config('configs/extr_results.json')
    config['open_from_saved'] = False
    config['save_dir'] = f'figures/{save_directory}'
    config['should_identify_thresholds'] = should_identify_thresholds
    for log_name in tqdm(os.listdir(f'logs/optimal_subsets/{log_directory}'), desc='Logs'):
        config['input_log_path'] = f'logs/optimal_subsets/{log_directory}/{log_name}'
        extr_results_main(config)

def run_multiple_optimize_hyperparams():
    config = read_config('configs/optim_hyperparam.json')
    config['n_iters'] = 60
    for arrhythmia in ['AF', 'LBBB', 'RBBB', 'IAVB', 'PAC', 'PVC', 'STD', 'STE']:
        config['arrhythmia_label'] = arrhythmia
        config['log_name'] = f'hyperparams/hyperparams_rf_{arrhythmia}'
        optimize_hyperparam_main(config)

if __name__ == "__main__":
    run_multiple_optimize_hyperparams()
    pass