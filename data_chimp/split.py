import csv
import math
import random
import itertools
import time
import os
from pathlib import Path

import pandas as pd
import numpy as np

RAW_PATH = './data_chimp/raw'
DATA_PATHS = ['./waveform', './spectrogram', './wav2vec2']
seeds = [0, 42, 3407]

for seed in seeds:
    for DATA_PATH in DATA_PATHS:
        random.seed(seed)
        np.random.seed(seed)

        start_time = time.time()

        # split by file
        SPLIT_PATH = '{}_split_{}'.format(DATA_PATH, seed)

        paths = Path(DATA_PATH).rglob('*.csv')
        paths = list(itertools.islice(paths, 100000))
        print('Read {} files'.format(len(paths)))

        data_with_files = []
        for path in paths:
            item = np.loadtxt(path, delimiter=',')
            data_with_files.append((path.stem, item))

        random.shuffle(data_with_files)   

        train_index = int(len(data_with_files) * 0.8)
        dev_index = int(len(data_with_files) * 0.9)

        train = data_with_files[:train_index]
        dev = data_with_files[train_index:dev_index]
        test = data_with_files[dev_index:]

        Path(SPLIT_PATH).mkdir(parents=True, exist_ok=True)

        np.save('{}/train.npy'.format(SPLIT_PATH), train, allow_pickle=True)
        np.save('{}/dev.npy'.format(SPLIT_PATH), dev, allow_pickle=True)
        np.save('{}/test.npy'.format(SPLIT_PATH), test, allow_pickle=True)

        print("--- %s seconds ---" % (time.time() - start_time))

# TODO

# split by ID

# df = pd.read_excel('{}/Pant hoots and phases_2.xlsx'.format(RAW_PATH), dtype={'Build-up ends': float, 'Climax ends': float}, na_values=' ')
# df = df.replace(np.nan, 0)
# df['File Name'] = df['File Name'].str.replace(';', '_')



