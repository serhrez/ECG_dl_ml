# Import parent folder
import sys
sys.path.append('..')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mlpython.utils.functions import read_config

config = read_config('make_figs/configs/triangle_corr.json')
data = pd.read_csv(config['features_path'])

# Take 100 rows from the data
data = data.sample(30, random_state=0)

mask = np.triu(np.ones_like(data.corr()))

dataplot = sns.heatmap(data.corr(), cmap="YlGnBu", annot=True, mask=mask)

plt.savefig(config['save_path'])
plt.show()

# Put debug breakpoint here
pass