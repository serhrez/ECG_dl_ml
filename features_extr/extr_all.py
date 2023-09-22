# Add the directory of the parent folder to the system path
import sys
sys.path.append('..')

from mlpython.utils.functions import read_config, read_description
import os
import scipy.io as sio
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.stats
import neurokit2 as nk
from biosppy.signals.ecg import ecg as ecg_biosppy
import pywt
from mlpython.utils.functions import resample_to_500hz, safe_np_calc, \
    safe_np_calc_mult, flatten_array, get_logger, save_log
from pyentrp import entropy as ent
from neurokit2.complexity.entropy_approximate import entropy_approximate
from neurokit2.complexity.entropy_sample import entropy_sample
from neurokit2.complexity.entropy_shannon import entropy_shannon
import warnings
import traceback
from joblib import Parallel, delayed
from time import time

def get_baselines(header, leads):
    baselines = np.zeros(len(leads))
    for i, l in enumerate(header.split('\n')):
        entries = l.split(' ')
        if i==0:
            num_leads = int(entries[1])
        elif i<=num_leads:
            current_lead = entries[-1]
            if current_lead in leads:
                j = leads.index(current_lead)
                try:
                    baselines[j] = float(entries[4].split('/')[0])
                except:
                    pass
        else:
            break
    return baselines

def extr_hea_features(hea_lines):
    age = hea_lines[13][7:].strip()
    sex = hea_lines[14][7:].strip()
    arrhythmias = hea_lines[15][6:].strip()
    are_multiple_arrhythmias = arrhythmias.find(',') != -1
    return [age, sex, arrhythmias, 
            are_multiple_arrhythmias]
HEA_COLS = ['age', 'sex', 'arrhythmias', 
             'are_multiple_arrhythmias']

def remove_nan_waves_onoffset(waves_onoffset):
    nans_in_t = np.isnan(waves_onoffset['ECG_T_Peaks']) | np.isnan(
        waves_onoffset['ECG_T_Onsets']) | np.isnan(
            waves_onoffset['ECG_T_Offsets'])
    nans_in_p = np.isnan(waves_onoffset['ECG_P_Peaks']) | np.isnan(
        waves_onoffset['ECG_P_Onsets']) | np.isnan(
            waves_onoffset['ECG_P_Offsets'])
    nans_in_r = np.isnan(waves_onoffset['ECG_R_Onsets']) | np.isnan(
        waves_onoffset['ECG_R_Offsets'])
    nan_locations = nans_in_t | nans_in_p | nans_in_r

    for k in waves_onoffset.keys():
        waves_onoffset[k] = (np.array(
            waves_onoffset[k])[~nan_locations]).astype(int)

def remove_nan_r_peaks(r_peaks, waves_onoffset):
    nans_in_wave_peaks = np.isnan(waves_onoffset['ECG_P_Peaks']) | np.isnan(
        waves_onoffset['ECG_T_Peaks'])
    new_r_peaks = r_peaks[~nans_in_wave_peaks]
    if len(new_r_peaks) > 6:
        return new_r_peaks
    else:
        return r_peaks

def extr_time_features(filtered_recording, signal_time_reference, waves_onoffset, heart_rate):
    ts = signal_time_reference
    # P Wave Duration:
    p_wave_durations = ts[waves_onoffset['ECG_P_Offsets']] - ts[
        waves_onoffset['ECG_P_Onsets']]

    # QRS Complex Durations
    qrs_complex_durations = ts[waves_onoffset['ECG_R_Offsets']] - ts[
        waves_onoffset['ECG_R_Onsets']]

    # PR interval
    pr_intervals = ts[waves_onoffset['ECG_R_Onsets']] - ts[
        waves_onoffset['ECG_P_Onsets']]

    # T-wave durations
    t_wave_durations = ts[waves_onoffset['ECG_T_Offsets']] - ts[
        waves_onoffset['ECG_T_Onsets']]

    # P Wave Amplitudes
    p_wave_amplitudes = filtered_recording[waves_onoffset['ECG_P_Peaks']]

    # T Wave Amplitudes
    t_wave_amplitudes = filtered_recording[waves_onoffset['ECG_T_Peaks']]

    # Compute Ratios
    # ==============

    # QRS / P Duration
    rat_qrs_p_duration = qrs_complex_durations / p_wave_durations

    # QRS / T Duration
    rat_qrs_t_duration = qrs_complex_durations / t_wave_durations

    features = []
    features.extend(safe_np_calc_mult(rat_qrs_p_duration, [np.mean, np.std]))
    features.extend(safe_np_calc_mult(rat_qrs_t_duration, [np.mean, np.std]))
    features.extend(safe_np_calc_mult(pr_intervals, [np.mean, np.std]))
    features.extend(safe_np_calc_mult(heart_rate,  [np.mean, np.std]))
    features.extend(safe_np_calc_mult(p_wave_amplitudes, [np.mean, np.std]))
    features.extend(safe_np_calc_mult(t_wave_amplitudes, [np.mean, np.std]))

    return features
TIME_FEATS = [[f'{feat}_mean', f'{feat}_std'] for feat in ['qrs_p_dur', 'qrs_t_dur', \
                'pr_interval', 'heart_rate', \
                'p_wave_amp', 't_wave_amp']]
TIME_FEATS = flatten_array(TIME_FEATS)

def extr_wavelet_features(filtered_recording):
    w = pywt.Wavelet("db1")
    features = []
    swt_lvl1 = pywt.swt(filtered_recording, w, 1)
    features.append(entropy_shannon(swt_lvl1[0][0])[0])
    return features
WAVELET_FEATS = ['swt_lvl1']

def extr_winner_features(waves_onoffset, signal_time_reference, r_peaks, heart_rate):
    ts = signal_time_reference
    p_waves = []

    for i in range(0, len(waves_onoffset['ECG_P_Onsets'])):
        time_idx_onset = waves_onoffset['ECG_P_Onsets'][i]
        time_idx_offset = waves_onoffset['ECG_P_Offsets'][i]
        p_waves.extend(ts[time_idx_onset:time_idx_offset])

    t_waves = []

    for i in range(0, len(waves_onoffset['ECG_T_Onsets'])):
        time_idx_onset = waves_onoffset['ECG_T_Onsets'][i]
        time_idx_offset = waves_onoffset['ECG_T_Offsets'][i]
        t_waves.extend(ts[time_idx_onset:time_idx_offset])

    # Entropy
    # ============
    SampEnP = entropy_sample(p_waves,
                             delay=1,
                             dimension=2,
                             r=0.2 * np.std(p_waves, ddof=1))
    ApEnP = entropy_approximate(p_waves,
                                delay=1,
                                dimension=2,
                                r=0.2 * np.std(p_waves, ddof=1))
    ApEnR = entropy_approximate(r_peaks,
                                delay=1,
                                dimension=2,
                                r=0.2 * np.std(r_peaks, ddof=1))
    PeEnT = ent.permutation_entropy(t_waves, order=3, delay=1, normalize=False)
    MPeEnT = ent.multiscale_permutation_entropy(t_waves, m=3, delay=1, scale=5)

    winner_features = []
    winner_features.append(safe_np_calc(heart_rate, np.min))  # min heart rate
    winner_features.append(
        safe_np_calc(MPeEnT, np.std))  # T wave multiscale permutation entropy std. dev
    winner_features.append(safe_np_calc(heart_rate, np.max))  # max heart rate
    winner_features.append(
        safe_np_calc(MPeEnT, np.median))  # T wave multiscale permutation entropy median

    winner_features.append(safe_np_calc(heart_rate, np.mean))  # Heart Rate µ

    winner_features.append(PeEnT)  # T wave permutation entropy std. dev
    winner_features.append(SampEnP[0])  # P wave sample entropy std. dev
    winner_features.append(ApEnP[0])  # Median P wave approximate entropy
    winner_features.append(ApEnR[0])  # R peak approximate entropy

    return winner_features
WINNER_FEATS = ['MIN_HR', 'T_MPE_STD', 'MAX_HR', 'T_MPE_MED', \
                'HR_MEAN', 'T_PE_STD', 'P_SE_STD', 'P_APE_MED', 'R_APE_MED']

def extr_RR_intervals_feats(r_peaks, signal_time_reference):
    rr_intervals = np.diff(signal_time_reference[r_peaks])
    delta_rr = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(delta_rr**2))
    nn60 = np.sum(np.abs(delta_rr) > 60)
    pNN60 = nn60 / len(rr_intervals) * 100

    return [rmssd, safe_np_calc(delta_rr, np.min), \
            pNN60, safe_np_calc(rr_intervals, np.median)]
RR_INTR_FEATS = ['rmssd', 'min_delta_rr', \
                 'pNN60', 'median_rr_interval']

def extract_heart_rate_variabilities(r_peaks,
                                      signal_frequency):
    hrv_features = []

    hrv_time = nk.hrv_time(r_peaks, sampling_rate=signal_frequency, show=False)
    hrv_features.extend(hrv_time.values[0])

    return hrv_features
HRV_FEATURES = ['HRV_MeanNN', 'HRV_SDNN', 'HRV_SDANN1', 'HRV_SDNNI1', 'HRV_SDANN2',\
       'HRV_SDNNI2', 'HRV_SDANN5', 'HRV_SDNNI5', 'HRV_RMSSD', 'HRV_SDSD',\
       'HRV_CVNN', 'HRV_CVSD', 'HRV_MedianNN', 'HRV_MadNN', 'HRV_MCVNN',\
       'HRV_IQRNN', 'HRV_Prc20NN', 'HRV_Prc80NN', 'HRV_pNN50', 'HRV_pNN20',\
       'HRV_MinNN', 'HRV_MaxNN', 'HRV_HTI', 'HRV_TINN']

def extr_sms_features(recording_lead, env):
    # Make recording_lead even length
    if len(recording_lead) % 2 != 0:
        recording_lead = recording_lead[:-1]
    signal_frequency = 500
    features = []
    try:
        (signal_time_reference, filtered_recording, r_peaks, template_ts,
                    templates, heart_rate_ts,
                    heart_rate) = ecg_biosppy(recording_lead, sampling_rate=500, show=False)
    except Exception as e:
        features.extend([np.nan] * len(SMS_FEATURES))
        return features

    try:
        wavelet_feats = extr_wavelet_features(filtered_recording)
        features.extend(wavelet_feats)
    except Exception as e:
        features.extend([np.nan] * len(WAVELET_FEATS))
    
    try: 
        with warnings.catch_warnings():
            # Ignore specific warning. This warning occurs because of the nk library.
            warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)
            # RuntimeWarning: invalid value encountered in double_scalars numpy
            warnings.simplefilter(action='ignore', category=RuntimeWarning)
            _, waves_onoffset = nk.ecg_delineate(filtered_recording,
                                                r_peaks,
                                                method="dwt",
                                                sampling_rate=signal_frequency,
                                                show=False,
                                                check=True)
    except Exception as e:
        features.extend([np.nan] * (len(SMS_FEATURES) - len(features)))
        return features
    
    r_peaks = remove_nan_r_peaks(r_peaks, waves_onoffset)
    remove_nan_waves_onoffset(waves_onoffset)

    try: 
        hrv_features = extract_heart_rate_variabilities(r_peaks, \
                                                        signal_frequency)
        features.extend(hrv_features)
    except Exception as e:
        features.extend([np.nan] * len(HRV_FEATURES))

    
    try: 
        rr_interval_features = extr_RR_intervals_feats(r_peaks, signal_time_reference)
        features.extend(rr_interval_features)
    except Exception as e:
        features.extend([np.nan] * len(RR_INTR_FEATS))
    
    try:
        time_features = extr_time_features(filtered_recording, signal_time_reference, \
                                            waves_onoffset, heart_rate)
        features.extend(time_features)
    except Exception as e:
        features.extend([np.nan] * len(TIME_FEATS))

    try:
        winner_features = extr_winner_features(waves_onoffset, signal_time_reference, \
                                                r_peaks, heart_rate)
        features.extend(winner_features)
    except Exception as e:
        features.extend([np.nan] * len(WINNER_FEATS))
    
    return features
SMS_FEATURES = WAVELET_FEATS + HRV_FEATURES + RR_INTR_FEATS + TIME_FEATS + WINNER_FEATS

def extr_frequency(hea_lines):
    return int(hea_lines[0].split(' ')[2])

def extract_features(hea_path, mat_path, env):
    with open(hea_path, 'r') as f:
        lines = f.readlines()
        hea_features = extr_hea_features(lines)
        frequency = extr_frequency(lines)
    # Read matlab file, extract relevant features
    mat = sio.loadmat(mat_path)['val'][:, :40000]
    # Resample signal
    mat = mat.astype(np.float64)
    mat = resample_to_500hz(mat, frequency)
    # Extract mat features
    features = []
    for lead_index in range(12):
        features.extend(extr_sms_features(mat[lead_index], env))
    # Create an array of age, sex, and ecg_signal features
    return hea_features + features
TOTAL_COLS = HEA_COLS + [f'{feat}_{i}' for i in range(1, 13) for feat in SMS_FEATURES]

def get_plain_list_of_records(description):
    records = []
    record_names_set = set()
    datasets = []
    for dataset in description.keys():
        for arrhythmia in description[dataset].keys():
            for record in description[dataset][arrhythmia]:
                if record in record_names_set:
                    continue
                record_names_set.add(record)
                records.append(record)
                datasets.append(dataset)
    return records, datasets    

# def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
#     log = file if hasattr(file,'write') else sys.stderr
#     traceback.print_stack(file=log)
#     log.write(warnings.formatwarning(message, category, filename, lineno, line))

# warnings.showwarning = warn_with_traceback

def main():
    LOG_TIMESTAMP = str(int(time()))
    log_dict = {}
    log_dict['start_time'] = time()
    config = read_config('configs/extract_feats_config.json')
    env = {}
    if config['supress_warnings'] == 'True':
        warnings.filterwarnings("ignore")
        warnings.simplefilter("ignore")

    description_path = os.path.join(config['dataset_path'], config['description_file_name'])
    description = read_description(description_path)

    records, datasets = get_plain_list_of_records(description)
    
    log_dict['number_of_records'] = len(records)
    log_dict['config'] = config

    def extract_features_by_index(index):
        record = records[index]
        dataset = datasets[index]

        hea_path = os.path.join(config['dataset_path'], record + '.hea')
        mat_path = os.path.join(config['dataset_path'], record + '.mat')
        if not os.path.exists(hea_path) or not os.path.exists(mat_path):
            return None
        feats = extract_features(hea_path, mat_path, env)

        return [dataset, record] + feats

    print("TOTAL RECORDS: ", len(records))
    if config['is_parallel'] == 'True':
        total_feats = Parallel(n_jobs=-1, verbose=1)(delayed(extract_features_by_index)(i) for i in range(len(records)))
    else:
        total_feats = []
        indices = list(range(len(records)))
        for i in indices:
            print("EXTRACTING RECORD: ", i)
            total_feats.append(extract_features_by_index(i))
    # Create dataframe with features and columns names
    df = pd.DataFrame(total_feats, columns=['dataset', 'record'] + TOTAL_COLS)
    # Save dataframe to csv file
    df.to_csv(config['extract_feats_path'], index=False)
    log_dict['end_time'] = time()
    save_log(log_dict, config['log_name'], LOG_TIMESTAMP)

if __name__ == '__main__':
    main()