from utils.snomed_ct import get_snomed, get_arrhythmia
from utils.functions import read_config, append_values_to_dictionary, \
    dictionary_perform, shuffle_data, under_sample_by_least, \
    save_log_np, identify_optimal_thresholds, one_hot_encode, \
    flatten_array, get_highly_correlated_features, \
    add_leads_to_feats_list, read_npy_log, append_one_val_to_dictionary
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score as sklearn_f1_score, roc_auc_score, \
                            accuracy_score, precision_score, recall_score
from dataset import Dataset, split_dataset
import importlib
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from utils.constants import LEAD_NAMES
from time import time
from scipy.stats import ttest_rel
import os
import warnings
from PIL import Image
warnings.filterwarnings("error")

COLORS_LIST = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', \
    'white', 'darkblue', 'darkgreen', 'darkred', 'deepskyblue', 'deeppink', \
    'gold', 'indigo', 'lime', 'maroon', 'navy', 'olive', 'purple', 'silver', \
    'teal', 'tomato'
]

HIGHLIGHT_COLOR = '#69ff1f'
HIGHLIGHT_COLOR2 = '#03fca5'
HIGHLIGHT_COLOR3 = '#03e8fc'

def save_table_as_file(table, file_path):
    np.save(file_path, table)

def get_lead_steps_la(input_log):
    lead_steps = []
    lead_steps_set = set()
    for lead_comb in input_log['report']['lead_steps'][::-1]:
        for lead in lead_comb:
            if lead not in lead_steps_set:
                lead_steps_set.add(lead)
                lead_steps.append(lead)
    return lead_steps

def get_p_values_la(total_f1_scores):
    best_lead = [0]
    p_values = [0]
    for i in range(1,len(total_f1_scores)):
        p_val = ttest_rel(total_f1_scores[best_lead[-1]], total_f1_scores[i]).pvalue

        if p_val < 0.05 and np.mean(total_f1_scores[i]) > np.mean(total_f1_scores[best_lead[-1]]):
            for j in range(best_lead[-1] + 1, i):
                p_val = ttest_rel(total_f1_scores[j], total_f1_scores[i]).pvalue
                if p_val > 0.05:
                    best_lead.append(j)
                    break
                if np.mean(total_f1_scores[j]) > np.mean(total_f1_scores[i]):
                    best_lead.append(j)
                    break
            else:
                best_lead.append(i)
        else:
            best_lead.append(best_lead[-1])
        p_values.append(p_val)
    return p_values, best_lead

def extr_classification_report(metrics_dict, probabilities, target, best_thresholds):
    if best_thresholds is not None:
        predicted = (probabilities >= best_thresholds).astype(int)
    else:
        predicted = probabilities
    target_argmax = np.argmax(target, axis=1)
    predicted_argmax = np.argmax(predicted, axis=1)
    append_values_to_dictionary(
                metrics_dict,
                classification_report(target_argmax, predicted_argmax, zero_division=0, output_dict=True))

def extr_metrics(metrics_dict, probabilities, target, best_thresholds):
    if best_thresholds is not None:
        predicted = (probabilities >= best_thresholds).astype(int)
    else:
        predicted = probabilities
    target_argmax = np.argmax(target, axis=1)
    predicted_argmax = np.argmax(predicted, axis=1)

    f1_score_macro = sklearn_f1_score(target_argmax, predicted_argmax, average='macro')
    f1_score_weighted = sklearn_f1_score(target_argmax, predicted_argmax, average='weighted')
    auc_macro = roc_auc_score(target, probabilities, multi_class='ovo', average='macro')
    auc_weighted = roc_auc_score(target, probabilities, multi_class='ovo', average='weighted')
    accuracy = accuracy_score(target_argmax, predicted_argmax)

    precision_macro = precision_score(target_argmax, predicted_argmax, zero_division=0, average='macro')
    precision_weighted = precision_score(target_argmax, predicted_argmax, zero_division=0, average='weighted')
    recall_macro = recall_score(target_argmax, predicted_argmax, zero_division=0, average='macro')
    recall_weighted = recall_score(target_argmax, predicted_argmax, zero_division=0, average='weighted')
    append_one_val_to_dictionary(
                metrics_dict, 
                f'f1_score_macro', 
                f1_score_macro)
    append_one_val_to_dictionary(
                metrics_dict,
                f'f1_score_weighted',
                f1_score_weighted)
    append_one_val_to_dictionary(
                metrics_dict,
                f'auc_macro',
                auc_macro)
    append_one_val_to_dictionary(
                metrics_dict,
                f'auc_weighted',
                auc_weighted)
    append_one_val_to_dictionary(
                metrics_dict,
                f'accuracy',
                accuracy)
    append_one_val_to_dictionary(
                metrics_dict,
                f'precision_macro',
                precision_macro)
    append_one_val_to_dictionary(
                metrics_dict,
                f'precision_weighted',
                precision_weighted)
    append_one_val_to_dictionary(
                metrics_dict,
                f'recall_macro',
                recall_macro)
    append_one_val_to_dictionary(
                metrics_dict,
                f'recall_weighted',
                recall_weighted)

def take_indices_in_dict(work_dict, indices):
    work_dict = work_dict.copy()
    for key in work_dict.keys():
        if type(work_dict[key]) is dict:
            work_dict[key] = take_indices_in_dict(work_dict[key], indices)
        else:
            work_dict[key] = [work_dict[key][i] for i in indices]
    return work_dict

def fill_mean_values_in_dict(work_dict, new_length):
    work_dict = work_dict.copy()
    for key in work_dict.keys():
        if type(work_dict[key]) is dict:
            work_dict[key] = fill_mean_values_in_dict(work_dict[key], new_length)
        else:
            old_length = len(work_dict[key])
            old_mean = np.mean(work_dict[key])
            work_dict[key].extend([old_mean] * (new_length - old_length))
            new_mean = np.mean(work_dict[key])
    return work_dict

def calc_metrics(input_log):
    cl_dicts = input_log['report']['classifier_dicts'][::-1]
    val_metrics = {}
    test_metrics = {}
    def do_work(index):
        val_current_metrics = {}
        test_current_metrics = {}
        val_classification_report = {}
        test_classification_report = {}
        for j in range(len(cl_dicts[index]['val_target_values'])):
            if input_log['should_identify_thresholds']:
                optimal_thresholds = identify_optimal_thresholds(
                    one_hot_encode(cl_dicts[index]['val_target_values'][0]), 
                    cl_dicts[index]['val_probabilities'][0])[0]
            else:
                optimal_thresholds = None
            extr_metrics(
                val_current_metrics, 
                cl_dicts[index]['val_probabilities'][j], 
                one_hot_encode(cl_dicts[index]['val_target_values'][j]),
                optimal_thresholds)
            extr_metrics(
                test_current_metrics, 
                cl_dicts[index]['test_probabilities'][j], 
                one_hot_encode(cl_dicts[index]['test_target_values']),
                optimal_thresholds)
            extr_classification_report(
                val_classification_report,
                cl_dicts[index]['val_probabilities'][j],
                one_hot_encode(cl_dicts[index]['val_target_values'][j]),
                optimal_thresholds)
            extr_classification_report(
                test_classification_report,
                cl_dicts[index]['test_probabilities'][j],
                one_hot_encode(cl_dicts[index]['test_target_values']),
                optimal_thresholds)
                    
        f1_scores_macro_val = val_current_metrics[f'f1_score_macro']
        f1_scores_macro_test = test_current_metrics[f'f1_score_macro']
        min_f1 = read_config(input_log['config']['minimum_f1_scores_path'])
        remove_indices = []

        for i in range(len(f1_scores_macro_val)):
            if f1_scores_macro_val[i] < min_f1:
                remove_indices.append(i)
        normal_indices = [i for i in range(len(f1_scores_macro_val)) if i not in remove_indices]

        val_current_metrics = take_indices_in_dict(val_current_metrics, normal_indices)
        test_current_metrics = take_indices_in_dict(test_current_metrics, normal_indices)
        val_classification_report = take_indices_in_dict(val_classification_report, normal_indices)
        test_classification_report = take_indices_in_dict(test_classification_report, normal_indices)

        val_current_metrics = fill_mean_values_in_dict(val_current_metrics, len(f1_scores_macro_val))
        test_current_metrics = fill_mean_values_in_dict(test_current_metrics, len(f1_scores_macro_val))
        val_classification_report = fill_mean_values_in_dict(val_classification_report, len(f1_scores_macro_val))
        test_classification_report = fill_mean_values_in_dict(test_classification_report, len(f1_scores_macro_val))

        temp_dict = {'val': f1_scores_macro_val, 'test': f1_scores_macro_test}
        temp_dict = take_indices_in_dict(temp_dict, normal_indices)
        temp_dict = fill_mean_values_in_dict(temp_dict, len(f1_scores_macro_val))
        f1_scores_macro_val = temp_dict['val']
        f1_scores_macro_test = temp_dict['test']            

        for metrics_dict in [val_current_metrics, test_current_metrics, \
                             val_classification_report, test_classification_report]:
            dictionary_perform(metrics_dict, np.mean)
        return val_current_metrics, test_current_metrics, \
               f1_scores_macro_val, f1_scores_macro_test, \
               val_classification_report, test_classification_report
    
    parallel = True
    if parallel:
        val_metrics_arr, test_metrics_arr, \
        f1_scores_macro_val, f1_scores_macro_test, \
        val_classification_report, test_classification_report = zip(*Parallel(n_jobs=-1)(delayed(do_work)(idx) for idx in range(len(cl_dicts))))
    else:
        val_metrics_arr, test_metrics_arr, \
        f1_scores_macro_val, f1_scores_macro_test, \
        val_classification_report, test_classification_report = [], [], [], [], [], []
        for idx in range(len(cl_dicts)):
            val_current_metrics, test_current_metrics, \
            f1_scores_macro_val_current, f1_scores_macro_test_current, \
            val_classification_report_current, test_classification_report_current = do_work(idx)
            val_metrics_arr.append(val_current_metrics)
            test_metrics_arr.append(test_current_metrics)
            f1_scores_macro_val.append(f1_scores_macro_val_current)
            f1_scores_macro_test.append(f1_scores_macro_test_current)
            val_classification_report.append(val_classification_report_current)
            test_classification_report.append(test_classification_report_current)
    
    for i in range(len(val_metrics_arr)):
        append_values_to_dictionary(val_metrics, val_metrics_arr[i])
        append_values_to_dictionary(test_metrics, test_metrics_arr[i])
    val_metrics['p_values'], val_metrics['optimal'] = get_p_values_la(f1_scores_macro_val)
    test_metrics['p_values'], test_metrics['optimal'] = get_p_values_la(f1_scores_macro_test)
    return {
        'val': val_metrics,
        'test': test_metrics,
        'val_classification_reports': val_classification_report,
        'test_classification_reports': test_classification_report
    }

def create_figure(output_dict, config, metrics_list):
    val_metrics_la = output_dict['metrics_la']['val']
    test_metrics_la = output_dict['metrics_la']['test']
    leads_range = list(range(1, len(output_dict['lead_steps_la']) + 1))

    # Creating the plot
    fig, ax = plt.subplots(figsize=(15, 15))

    def plot_all_metrics(metrics_dict, metrics_list, prefix, marker, colors):
        for (i, metric) in enumerate(metrics_list):
            ax.plot(leads_range, metrics_dict[metric], marker, label=prefix + metric, color=colors[i])
    
    plot_all_metrics(val_metrics_la, metrics_list, 'val_', 'o-', COLORS_LIST)
    plot_all_metrics(test_metrics_la, metrics_list, 'test_', 's--', COLORS_LIST)

    ax.set_xlabel('Number of leads')
    ax.set_ylabel('Metric value')
    ax.set_xticks(leads_range)
    ax.set_xticklabels([f"{num}:{lead}"for lead, num in zip(output_dict['lead_steps_la'], leads_range)])
    
    # Adding a legend to the plot
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

    plt.tight_layout()
    plt.savefig(os.path.join(config['save_dir'], f'all_metrics_{output_dict["arrhythmia_label"]}.png'), dpi=500)
    plt.close()  # Close the figure to prevent it from displaying

def create_val_testing_figure(output_dict, config):
    optimal_num_of_leads = max(output_dict['metrics_la']['val']['optimal'][-1], 
                                 output_dict['metrics_la']['test']['optimal'][-1]) + 1
    val_metrics_la = output_dict['metrics_la']['val']['f1_score_macro']
    test_metrics_la = output_dict['metrics_la']['test']['f1_score_macro']
    leads_range = list(range(1, len(output_dict['lead_steps_la']) + 1))
    fig, ax = plt.subplots(figsize=(5 * 2.401408450704225, 5))
    ax.plot(leads_range, val_metrics_la, 'o-', label='Validation')
    ax.plot(leads_range, test_metrics_la, 'o-', label='Testing')
    ax.legend(loc='upper left')
    ax.set_xticks(leads_range)
    ax.set_xlim(1 - 0.1, len(leads_range) + 0.1)
    if optimal_num_of_leads == 12:
        optimal_num_of_leads += 0.1
    ax.set_xticklabels([""] * len(leads_range))
    ax.tick_params(axis='both', direction='in')
    ax.axvspan(1 - 0.1, optimal_num_of_leads, facecolor='green', alpha=0.2)

    plt.savefig(os.path.join(config['save_dir'], f'fig_{output_dict["arrhythmia_label"]}.png'), dpi=500)
    plt.close()

def open_img_and_remove_borders(img_path, border):
    image = Image.open(img_path)
    image = image.crop((border, border, image.size[0] - border, image.size[1] - border))
    # Save image by path
    image.save(img_path)

def create_needed_table(output_dict, config):
    val_metrics_la = output_dict['metrics_la']['val']
    test_metrics_la = output_dict['metrics_la']['test']
    optimal_num_of_leads = max(output_dict['metrics_la']['val']['optimal'][-1], 
                                output_dict['metrics_la']['test']['optimal'][-1]) + 1

    # Creating the table data
    table_data = [
        output_dict['lead_steps_la']
    ]
    # ['mean F1-score on validation set']
    table_data.append([round(val, 3) for val in val_metrics_la['f1_score_macro']])
    # ['p-value of accuracy variation between number of leads used on validation set']
    table_data.append([round(val, 3) for val in val_metrics_la['p_values']])
    # ['mean F1-score on test set']
    table_data.append([round(val, 3) for val in test_metrics_la['f1_score_macro']])
    # ['p-value of accuracy variation between number of leads used on test set']
    table_data.append([round(val, 3) for val in test_metrics_la['p_values']])

    cellColors = []
    for i in range(len(table_data)):
        cellColors.append(['#FFFFFF'] * len(table_data[i]))
    for i in range(len(table_data)):
        for j in range(optimal_num_of_leads):
            cellColors[i][j] = "#D2E7D2"

    # Creating the table axes
    table_ax = plt.table(
        cellText=table_data,
        bbox=(0, 0, 1, 1),
        cellLoc='center',
        rowLoc='center',
        cellColours=cellColors
    )

    # Modifying the table properties
    table_ax.auto_set_font_size(False)
    table_ax.set_fontsize(20)
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    fig = plt.gcf()
    fig.set_size_inches(20, 5.785714285714286)  # Adjust the width and height as needed
    plt.tight_layout()
    table_path = os.path.join(config['save_dir'], f'needed_table_{output_dict["arrhythmia_label"]}.png')
    plt.savefig(table_path, dpi=500)
    plt.close()
    open_img_and_remove_borders(table_path, 74)
    save_table_as_file(table_data, os.path.join(config['save_dir'], f'npy/needed_table_{output_dict["arrhythmia_label"]}.npy'))

def create_table(output_dict, config, metrics_list):
    val_metrics_la = output_dict['metrics_la']['val']
    test_metrics_la = output_dict['metrics_la']['test']

    # Creating the table data
    table_data = [
        ['Metric'] + output_dict['lead_steps_la'],
    ]
    cellColors = []
    cellColors.append(['#FFFFFF'] * len(table_data[-1]))

    for metric in metrics_list:
        table_data.append(['val_' + metric] + [round(val, 3) for val in val_metrics_la[metric]])
        table_data.append(['test_' + metric] + [round(val, 3) for val in test_metrics_la[metric]])
        
        cellColors.append(['#FFFFFF'] * (len(table_data[-1])))
        cellColors.append(['#FFFFFF'] * (len(table_data[-1])))
        cellColors[-2][np.argmax(val_metrics_la[metric]) + 1] = HIGHLIGHT_COLOR
        cellColors[-1][np.argmax(test_metrics_la[metric]) + 1] = HIGHLIGHT_COLOR

    # Add optimal label to table_data
    for metrics_la, metrics_label in zip([val_metrics_la, test_metrics_la], ['val_', 'test_']):
        table_data.append([f'{metrics_label}optimal'] + [output_dict['lead_steps_la'][i] for i in metrics_la['optimal']])
        cellColors.append(['#FFFFFF'] * (len(table_data[-1])))
        for i in range(metrics_la['optimal'].index(metrics_la['optimal'][-1]) + 1):
            if i == 0 or metrics_la['optimal'][i] != metrics_la['optimal'][i - 1]:
                cellColors[-1][i + 1] = HIGHLIGHT_COLOR3
        table_data.append([f'{metrics_label}p_values'] + [round(val, 3) for val in metrics_la['p_values']])
        cellColors.append(['#FFFFFF'] * (len(table_data[-1])))
        for i in range(len(metrics_la['p_values'])):
            if metrics_la['p_values'][i] < 0.05:
                cellColors[-1][i + 1] = HIGHLIGHT_COLOR2
        final_optimal = [output_dict['lead_steps_la'][i] for i in range(metrics_la['optimal'][-1] + 1)]
        table_data.append(
            [f'{metrics_label}final_subset'] +\
            final_optimal + \
            [''] * (len(output_dict['lead_steps_la']) - len(final_optimal)))
        cellColors.append(['#FFFFFF'] * (len(table_data[-1])))

    colwidths = [0.4] + [0.1] * (len(table_data[0]) - 1)
    # Make colwidths sum to be 1
    colwidths = [round(val / sum(colwidths), 2) for val in colwidths]
    # Creating the table axes
    table_ax = plt.table(
        cellText=table_data, 
        loc='center', 
        colWidths=colwidths,
        cellLoc='center',
        rowLoc='center',
        cellColours=cellColors
    )

    # Modifying the table properties
    table_ax.auto_set_font_size(False)
    table_ax.set_fontsize(12)
    table_ax.scale(1, 1.1)
    plt.axis('off')

    fig = plt.gcf()
    fig.set_size_inches(20, 4.984709480122324)  # Adjust the width and height as needed

    plt.savefig(os.path.join(config['save_dir'], f'all_metrics_table_{output_dict["arrhythmia_label"]}.png'), dpi=500)
    plt.close()
    save_table_as_file(table_data, os.path.join(config['save_dir'], f'npy/all_metrics_table_{output_dict["arrhythmia_label"]}.npy'))

def create_table_percentages(output_dict, config):
    def get_percentages(f1_scores):
        max_f1_score = max(f1_scores)
        differences = [round((1 - val / max_f1_score) * 100, 3) for val in f1_scores]
        percentages = []
        for diff in differences:
            if diff == 0:
                percentages.append("0%")
            else:
                percentages.append(f"-{diff}%")
        return percentages
    table_data = [
        [""] + output_dict['lead_steps_la'],
        ["Val:F1-scores"] + [round(val, 3) for val in output_dict['metrics_la']['val']['f1_score_macro']],
        ["Val:Percentages"] + get_percentages(output_dict['metrics_la']['val']['f1_score_macro']),
        ["Test:F1-scores"] + [round(val, 3) for val in output_dict['metrics_la']['test']['f1_score_macro']],
        ["Test:Percentages"] + get_percentages(output_dict['metrics_la']['test']['f1_score_macro'])
    ]

    colwidths = [0.2] + [0.1] * (len(table_data[0]) - 1)
    # Make colwidths sum to be 1
    colwidths = [round(val / sum(colwidths), 2) for val in colwidths]
    # Creating the table axes
    table_ax = plt.table(
        cellText=table_data, 
        loc='center', 
        colWidths=colwidths,
        cellLoc='center',
        rowLoc='center'
    )

    # Modifying the table properties
    table_ax.auto_set_font_size(False)
    table_ax.set_fontsize(12)
    table_ax.scale(1.2, 1.3)
    plt.axis('off')

    fig = plt.gcf()
    fig.set_size_inches(len(table_data[0]) + 1, len(table_data) + 1)  # Adjust the width and height as needed

    plt.savefig(os.path.join(config['save_dir'], f'percentages_{output_dict["arrhythmia_label"]}.png'), dpi=500)
    plt.close()

    save_table_as_file(table_data, os.path.join(config['save_dir'], f'npy/percentages_{output_dict["arrhythmia_label"]}.npy'))

def make_plots(output_dict, config):
    all_metrics = [
        'f1_score_macro', 
        'f1_score_weighted', 
        'auc_macro', \
        'auc_weighted', 
        'accuracy', 
        'precision_macro', \
        'precision_weighted', 
        'recall_macro', 
        'recall_weighted']
    non_weighted_metrics = [
        metric for metric in all_metrics if \
        'weighted' not in metric]

    create_figure(output_dict, config, non_weighted_metrics)
    create_val_testing_figure(output_dict, config)
    create_table(output_dict, config, all_metrics)
    create_table_percentages(output_dict, config)
    create_needed_table(output_dict, config)

def create_output_dict(input_log):
    output_dict = {}
    output_dict['arrhythmia_label'] = input_log['config']['arrhythmia_label']
    output_dict['labels_order'] =  [input_log['config']['arrhythmia_label'], 'NSR', 'Other']
    # _la - lead added. From lowest number of lead to the highest
    output_dict['f1_scores_la'] = input_log['report']['f1_scores'][::-1]
    output_dict['lead_steps_la'] = get_lead_steps_la(input_log)
    metrics_la = calc_metrics(input_log)
    output_dict['metrics_la'] = metrics_la

    return output_dict

def main(config):
    input_log = read_npy_log(config['input_log_path'])
    input_log['config']['minimum_f1_scores_path'] = 'deeplearning/with_feats_extr/configs/minimum_f1_scores.json'

    if 'should_identify_thresholds' in config:
        input_log['should_identify_thresholds'] = config['should_identify_thresholds']
    else:
        input_log['should_identify_thresholds'] = True

    if config['open_from_saved']:
        output_dict = np.load(config['output_dict_path'], allow_pickle=True).item()
    else:
        output_dict = create_output_dict(input_log)
    # Save output_dict to temporary location:
    np.save(config['output_dict_path'], output_dict)
    make_plots(output_dict, config)
    pass

if __name__ == "__main__":
    config = read_config('configs/extr_results.json')
    main(config)