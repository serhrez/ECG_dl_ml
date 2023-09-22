import os
import json

def create_hyperparam_config():
    all_arhythmias = ['AF', 'LBBB', 'RBBB', 'IAVB', 'PAC', 'PVC', 'STD', 'STE']
    parameters = {}
    for logfile in os.listdir('logs/hyperparams'):
        for arrhythmia in all_arhythmias:
            if f'rf_{arrhythmia}' in logfile:
                hyperparams_log_dict = json.load(open(f'logs/hyperparams/{logfile}', 'r'))
                parameters[arrhythmia] = hyperparams_log_dict['best_params']
                parameters[f"{arrhythmia}_optimal_thresholds"] = hyperparams_log_dict['optimal_thresholds']
    json.dump(parameters, open('configs/hyperparams.json', 'w'), indent=4)

if __name__ == "__main__":
    create_hyperparam_config()