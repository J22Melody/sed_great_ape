import csv
import math
import random
import itertools
import time
import os
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torchaudio

seed = 42
random.seed(seed)
torch.manual_seed(0)
np.random.seed(0)

start_time = time.time()

WAV_PATH = './data_chimp/raw'
DATA_PATH = './data_chimp/wav2vec2'

# split by file

SPLIT_PATH = './data_chimp/wav2vec2_split_by_file'

paths = Path(DATA_PATH).rglob('*.csv')
paths = list(itertools.islice(paths, 100000))
# print('Read {} files: {}'.format(len(paths), paths))
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

# split by file (small split)

# TODO

# split by ID

# df = pd.read_excel('{}/Pant hoots and phases_2.xlsx'.format(WAV_PATH), dtype={'Build-up ends': float, 'Climax ends': float}, na_values=' ')
# df = df.replace(np.nan, 0)
# df['File Name'] = df['File Name'].str.replace(';', '_')



