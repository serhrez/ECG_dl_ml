from leads_feat_extr import main as leads_feat_extr_main
from utils.functions import read_config
from dl_optimize_hyperparam import main as dl_optimize_hyperparam_main
from dl_optimal_subsets import main as dl_optimal_subsets_main
from tqdm import tqdm
import json

DIR_PATH = "deeplearning/with_feats_extr"

def extr_multiple_leads_feats():
    leads = [i for i in range(12)]
    arrhythmias = ['AF', 'LBBB', 'RBBB', 'IAVB', 'PAC', 'PVC', 'STD', 'STE']
    config = read_config(f'{DIR_PATH}/configs/leads_feat_extr.json')
    for arrhythmia in arrhythmias:
        for lead in leads:
            print("=" * 20)
            print("Extract features for arrhythmia: ", arrhythmia, " and lead: ", lead)
            config['arrhythmia_label'] = arrhythmia
            config['lead'] = lead
            leads_feat_extr_main(config)

def mult_optimize_hyperparams():
    config = read_config(f'{DIR_PATH}/configs/optimize_hyperparams.json')
    for num_lead in tqdm(range(1, 13), desc='Leads'):
        config['number_of_leads'] = num_lead
        dl_optimize_hyperparam_main(config)

def combine_hyperparams(arrhythmia_label):
    all_hyperparams = {}
    for lead in range(12):
        hyperparams_path = f"{DIR_PATH}/hyperparameters_logs/{arrhythmia_label}_{lead + 1}.json"
        with open(hyperparams_path, 'r') as f:
            hyperparams = json.load(f)
        best_hyperparams = hyperparams['best_hyperparams'][0]
        all_hyperparams[lead] = best_hyperparams
    with open(f"{DIR_PATH}/configs/hyperparams.json", 'w') as f:
        json.dump(all_hyperparams, f, indent=2)

def mult_optimal_subsets():
    config = read_config(f'{DIR_PATH}/configs/dl_optimal_subsets.json')
    arrhythmias = ['STE']
    for i in tqdm(range(len(arrhythmias)), desc='Mult optimal subsets'):
        arrhythmia_label = arrhythmias[i]
        config['arrhythmia_label'] = arrhythmia_label
        config['log_path'] = f'optimal_subsets/dl4/{arrhythmia_label}'
        dl_optimal_subsets_main(config)

if __name__ == "__main__":
    # extr_multiple_leads_feats()
    # mult_optimize_hyperparams()
    # combine_hyperparams('AF')
    mult_optimal_subsets()