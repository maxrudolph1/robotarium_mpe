import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import json
import numpy as np
import matplotlib.patches as mpatches

f = open('data_dict.json')
data = json.load(f)

meth_list = list(data.keys())
env_list = list(data[meth_list[0]].keys())

for meth in meth_list:
    for env in env_list:
        print(np.array(data[meth][env]).shape)