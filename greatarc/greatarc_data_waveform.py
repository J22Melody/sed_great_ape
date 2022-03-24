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

TXT_PATH = './data_greatarc/wetransfer_audio-files-for-zifan_2021-12-14_1301'
WAV_PATH = './data_greatarc/wetransfer_audio-files-for-zifan_2021-12-14_1301'

def parse_raw():
    call_type_dict = {
        'Grumph': 1,
        'Kiss-squeak': 2,
        'Rolling call': 3,
    }

    wavpaths = Path(WAV_PATH).rglob('*.wav')
    for i, wavpath in enumerate(wavpaths):
        wavpath = str(wavpath)
        filename = wavpath.split('/')[-1].split('.')[0]
        # check whether annotations exist
        txtpath = '{}/{}.Table.1.selections.txt'.format(TXT_PATH, filename)
        if not os.path.exists(txtpath):
            continue

        print('processing {} ...'.format(wavpath))

        # read wav file
        waveform, sample_rate = torchaudio.load(wavpath)

        # transform
        new_sample_rate = 8000
        transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
        transformed = transform(waveform)

        # reshape
        length = transformed.size()[1]
        new_length = math.ceil(length / new_sample_rate) * new_sample_rate
        target = torch.zeros(1, new_length)
        target[:, :length] = transformed
        target = target.view(-1, new_sample_rate)

        # read txt annotations
        calls = None
        with open(txtpath, newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            data = list(reader)[1:]
            calls = [(float(item[3]), float(item[4]), call_type_dict.get(item[-3].rstrip('?'), 4)) for item in data]

        # create y
        y = []
        for i in range(target.size()[0]):
            y_ = 0
            t_begin = i
            t_end = i + 1
            for begin, end, call_type in calls:
                if not (t_begin > end or t_end < begin):
                    y_ = call_type
                    break
            y.append(y_)
        y = torch.tensor(y).view(-1, 1)

        data = torch.cat((y, target), dim=1)

        np.savetxt('./data_greatarc_waveform_1/{}.csv'.format(filename), data.numpy(), delimiter=',')

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

# parse_raw()
data_split()