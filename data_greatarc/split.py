import csv
import math
import random
import itertools
import time
import os
from pathlib import Path

import numpy as np
import torch
import torchaudio

from sklearn.utils import resample
from sklearn.model_selection import train_test_split

seed = 42
random.seed(seed)
torch.manual_seed(0)
np.random.seed(0)

start_time = time.time()

TXT_PATH = './raw/LC rhythm - all LC Tuanan/Raven tables'
WAV_PATH = './raw/LC rhythm - all LC Tuanan'

def data_split():
    train_paths = [
        'ElisaEtek experiment Blue sheet 6-1-2011',
        'YetYeni experiment tiger sheet 10-02-2011 (1)_1st half',
        'YetYeni experiment tiger sheet 10-02-2011 (3)',
        'Kelly experiment Tiger 19-12-2010',
        'Kelly experiment spots 11-1-2011',
    ]
    # test_dev_paths = [
    #     'Kelly experiment Tiger 19-12-2010',
    #     'Kelly experiment spots 11-1-2011',
    # ]

    train = np.concatenate([np.loadtxt('./data_greatarc_waveform_1/{}.csv'.format(path), delimiter=',') for path in train_paths])
    # test_dev = np.concatenate([np.loadtxt('./data_greatarc_waveform_1/{}.csv'.format(path), delimiter=',') for path in test_dev_paths])
    
    # train = train[train[:, 0] != 0]
    # test_dev = test_dev[test_dev[:, 0] != 0]
    
    train, test_dev = train_test_split(train, test_size=0.2, random_state=seed)
    dev, test = train_test_split(test_dev, test_size=0.5, random_state=seed)

    # downsampling
    # train_1 = train[train[:, 0] == 1]
    # train_0 = train[train[:, 0] == 0]
    # train_0 = resample(train_0, random_state=seed, n_samples=train_1.shape[0], replace=False)
    # train = np.concatenate([train_0, train_1])

    unique, counts = np.unique(train[:, 0], return_counts=True)
    counts_by_id = dict(zip(unique, counts))
    print('Labels in train before upsampling: ', counts_by_id)
    unique, counts = np.unique(dev[:, 0], return_counts=True)
    counts_by_id = dict(zip(unique, counts))
    print('Labels in dev before upsampling: ', counts_by_id)
    unique, counts = np.unique(test[:, 0], return_counts=True)
    counts_by_id = dict(zip(unique, counts))
    print('Labels in test before upsampling: ', counts_by_id)

    # upsampling
    # upsample_classes = [1, 2, 3, 4]
    # upsample_n = max(counts_by_id.values())
    # upsample_list = []
    # for i in upsample_classes:
    #     train_i = train[train[:, 0] == i]
    #     train_i = resample(train_i, random_state=seed, n_samples=upsample_n, replace=True)
    #     upsample_list.append(train_i)
    # train = np.concatenate(upsample_list)

    np.random.shuffle(train)

    print('Train:', train.shape)
    print('Dev:', dev.shape)
    print('Test:', test.shape)

    np.savetxt('./data_greatarc_waveform_1_split/train.csv', train, delimiter=',')
    np.savetxt('./data_greatarc_waveform_1_split/dev.csv', dev, delimiter=',')
    np.savetxt('./data_greatarc_waveform_1_split/test.csv', test, delimiter=',')

    print("--- %s seconds ---" % (time.time() - start_time))

data_split()