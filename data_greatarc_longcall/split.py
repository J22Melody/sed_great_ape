import csv
import math
import random
import itertools
import time
import os
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
import torch
import torchaudio

seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

start_time = time.time()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_PATH = './data_wav2vec2'
TXT_PATH = '../data_greatarc/wetransfer_long-call-recordings-raven-acoustics_2021-11-22_1051/LC rhythm - all LC Tuanan/Raven tables'
WAV_PATH = '../data_greatarc/wetransfer_long-call-recordings-raven-acoustics_2021-11-22_1051/LC rhythm - all LC Tuanan'

def data_split():
    paths = Path(DATA_PATH).rglob('*.csv')
    paths = list(itertools.islice(paths, 99999))
    print('Split {} files: {}'.format(len(paths), paths))
    data = [np.loadtxt(path, delimiter=',') for path in paths]

    data = np.concatenate(data)
    print(data.shape)

    unique, counts = np.unique(data[:, 0], return_counts=True)
    print('Labels in train: ', dict(zip(unique, counts)))

    print("--- %s seconds ---" % (time.time() - start_time))

data_split()