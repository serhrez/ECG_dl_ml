from utils.snomed_ct import get_snomed, get_arrhythmia
from utils.functions import read_config, add_values_to_dictionary, \
    dictionary_perform, shuffle_data, under_sample_by_least, \
    save_log_np

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score as sklearn_f1_score
from dataset import Dataset, split_dataset
import importlib
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from utils.functions import identify_optimal_thresholds, one_hot_encode
from scipy.optimize import differential_evolution
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight 
RF_NAME = 'random_forest'
GB_NAME = 'xgb_classifier'

def get_classifier(classifier_name):
    if classifier_name == RF_NAME:
        return RandomForestClassifier(class_weight='balanced')
    if classifier_name == GB_NAME:
        return XGBClassifier()

def get_search_distrib(classifier_name):
    if classifier_name == RF_NAME:
        n_estimators = [100, 110, 120, 130, 140, 150, 170, 190, 200, 220]
        max_depth = [30, 40, 50, 60, 70, 90, 110]
        min_samples_split = [2]
        min_samples_leaf = [2]
        bootstrap = [True]
        return {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'bootstrap': bootstrap
        }
    if classifier_name == GB_NAME:
        learning_rates = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1]
        n_estimators = [70, 100, 140, 200, 300, 500]
        max_depth = [3, 5, 7]
        return {
            'learning_rate': learning_rates,
            'n_estimators': n_estimators,
            'max_depth': max_depth
        }


def find_optimal_thresholds(df, classifier_name, labels_order, params, maxiter=300):
    def objective_function(thresholds, y_true, y_scores):
        y_pred = (y_scores >= thresholds).astype(int)
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_true, axis=1)
        return -sklearn_f1_score(y_true, y_pred, average='macro')
    
    (X, _), Y = df.get_X(), df.get_Y(labels_order)
    X, Y = shuffle_data(X, Y, random_state=0)
    if classifier_name == RF_NAME:
        cls = RandomForestClassifier(**params, class_weight='balanced')
    else:
        cls = XGBClassifier(**params)
    # Split data into train and validation
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=0)
    if classifier_name == RF_NAME:
        cls.fit(X_train, Y_train)
    else:
        classes_weights = compute_sample_weight(
            class_weight='balanced',
            y=Y_train
        )
        cls.fit(X_train, Y_train, sample_weight=classes_weights)
    Y_val_pred_proba = cls.predict_proba(X_val)
    
    y_true = one_hot_encode(Y_val)
    y_scores = Y_val_pred_proba

    bounds = [(0.0, 1.0)] * y_true.shape[1]  # Bounds for threshold values

    result = differential_evolution(
        objective_function,
        bounds,
        args=(y_true, y_scores),
        strategy='best1bin',
        popsize=10,
        tol=1e-3,
        recombination=0.7,
        mutation=(0.5, 1),
        maxiter=maxiter,
        seed=42
    )
    
    return result.x

def main(config):
    log_dict = {}
    log_dict['config'] = config
    arrhythmia_label = config['arrhythmia_label']
    labels_order = [arrhythmia_label, 'NSR', 'Other']

    dataset = Dataset(config['dataset_path'], config['description_file_name'])
    dataframe = dataset.get_pandas_dataframe(arrhythmia_label)

    features_df = pd.read_csv(config['features_path'])

    ds_module = importlib.import_module(config['ds_class_module'])
    ds_class = getattr(ds_module, config['ds_class_name'])

    df = ds_class(dataframe, features_df)

    (X, columns), Y = df.get_X(), df.get_Y(labels_order)

    # Shuffle data
    X, Y = shuffle_data(X, Y, random_state=0)
    X, Y = X[:config['n_samples']], Y[:config['n_samples']]

    log_dict['X_shape'] = X.shape
    classifier = get_classifier(config['classifier_name'])
    search_distrib = get_search_distrib(config['classifier_name'])

    log_dict['search_distrib'] = search_distrib

    random_search = RandomizedSearchCV(
        estimator=classifier,
        param_distributions=search_distrib,
        cv=3,
        verbose=2,
        n_jobs=-1,
        n_iter=config['n_iters']
    )
    random_search.fit(X, Y)
    log_dict['best_params'] = random_search.best_params_
    log_dict['best_score'] = random_search.best_score_
    log_dict['cv_results'] = random_search.cv_results_
    log_dict['cv_results_best_index'] = random_search.best_index_
    log_dict['optimal_thresholds'] = find_optimal_thresholds(df, config['classifier_name'], labels_order, random_search.best_params_)
    save_log_np(log_dict, config['log_name'])

if __name__ == '__main__':
    config = read_config('configs/optim_hyperparam.json')
    main()