import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
for _ in range(2):
    current = os.path.dirname(current)
sys.path.append(current)

from utils.constants import ARRHYTHMIA_LABELS
from dl_utils import get_whole_dataset
from models import get_extr_feats_model_multiple_feats_extr
from utils.functions import read_config, get_dataset_df, get_all_records
from keras.models import Model
import numpy as np
from tqdm import tqdm

DIR_PATH = "deeplearning/with_feats_extr"

def main(config):
    data, record_names = get_whole_dataset(config, [i for i in range(12)])
    data = np.squeeze(data)
    for arrhythmia in tqdm(ARRHYTHMIA_LABELS, desc='arrhythmia', leave=False):
        features = np.empty((data.shape[0], 12, 32))
        for lead in tqdm(range(12), desc='leads', leave=False):
            print("")
            model = get_extr_feats_model_multiple_feats_extr(3, config['record_len'])
            model.load_weights(f'{config["features_models"]}/{arrhythmia}_{lead}.h5')
            model_32 = Model(model.input, model.layers[-5].output)
            features[:, lead, :] = model_32.predict(data[:, :, lead])
        features_dict = {label: row for label, row in zip(record_names, features)}
        np.save(f'{config["features_save_path"]}/{arrhythmia}.npy', features_dict)
    pass

if __name__ == "__main__":
    config = read_config(f'{DIR_PATH}/configs/extr_all_feats.json')
    main(config)