import os
import numpy as np

def print_dictionary_structure(dictionary, indent=0):
    for key, value in dictionary.items():
        if isinstance(value, list):
            print("  " * indent + str(key) + " (list)")
            if len(value) > 0:
                if isinstance(value[0], dict):
                    print_dictionary_structure(value[0], indent + 1)
        elif isinstance(value, dict):
            print("  " * indent + str(key))
            print_dictionary_structure(value, indent + 1)
        else:
            print("  " * indent + str(key))

if __name__ == "__main__":
    dl_path = 'logs/optimal_subsets/dl4'
    new_dl_path = 'logs/optimal_subsets/dl4_converted'
    new_report = {}
    for file_name in os.listdir(dl_path):
        dl_report = np.load(f'{dl_path}/{file_name}', allow_pickle=True).item()
        rf_report = np.load(f'logs/optimal_subsets/rf/os_rec_rf_AF+1689931778.npy', allow_pickle=True).item()
        
        new_report['config'] = dl_report['config']
        new_report['duration'] = dl_report['duration']
        new_report['report'] = {}
        
        # Copy report:
        new_report['report']['f1_scores'] = dl_report['report']['f1_scores']
        new_report['report']['lead_feature_importances'] = dl_report['report']['lead_feature_importances']
        new_report['report']['lead_steps'] = dl_report['report']['lead_steps']

        # Copy classifier_dicts:
        new_report['report']['classifier_dicts'] = []
        for i in range(len(dl_report['report']['classifier_dicts'])):
            new_report['report']['classifier_dicts'].append({})
            for key in ['val', 'test']:
                new_report['report']['classifier_dicts'][i][f'{key}_probabilities'] = \
                    dl_report['report']['classifier_dicts'][i][key]['prob']
                new_report['report']['classifier_dicts'][i][f'{key}_predictions'] = \
                    dl_report['report']['classifier_dicts'][i][key]['pred']
                targets = dl_report['report']['classifier_dicts'][i][key]['target']
                if key == 'test':
                    targets = targets[0]
                new_report['report']['classifier_dicts'][i][f'{key}_target_values'] = targets
                    
        np.save(f'{new_dl_path}/{file_name}', new_report)