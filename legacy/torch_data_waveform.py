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

seed = 42
random.seed(seed)
torch.manual_seed(0)
np.random.seed(0)

start_time = time.time()

TXT_PATH = './data_full/Klaus_Zuberbuhler_1994_2002/Raven Pro Tables'
WAV_PATH = './data_full/Klaus_Zuberbuhler_1994_2002/Recordings'

def parse_raw():
    wavpaths = Path(WAV_PATH).rglob('*.wav')
    for i, wavpath in enumerate(wavpaths):
        wavpath = str(wavpath)
        filename = wavpath.split('/')[-1].split('.')[0]
        # check whether annotations exist
        txtpath = '{}/{}.txt'.format(TXT_PATH, filename)
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
            data = list(reader)
            calls = [(float(item[3]), float(item[4])) for item in data[1:]]

        # create y
        y = []
        for i in range(target.size()[0]):
            is_call = 0
            t_begin = i
            t_end = i + 1
            for begin, end in calls:
                if not (t_begin > end or t_end < begin):
                    is_call = 1
                    break
            y.append(is_call)
        y = torch.tensor(y).view(-1, 1)

        data = torch.cat((y, target), dim=1)

        # np.savetxt('./data_waveform_1/{}.csv'.format(filename), data.numpy(), delimiter=',')

def data_split():
    paths = Path('./data_waveform_1').rglob('*.csv')
    paths = list(itertools.islice(paths, 999999))
    print('Split {} files: {}'.format(len(paths), paths))
    data = [np.loadtxt(path, delimiter=',') for path in paths]
    random.shuffle(data)

    total_size = len(data)
    train_size = math.floor(total_size*0.8)
    dev_size = math.floor(total_size*0.1)
    test_size = total_size - train_size - dev_size

    print(train_size, dev_size, test_size)

    dev = np.concatenate(data[:dev_size])
    test = np.concatenate(data[dev_size:test_size + dev_size])
    train = np.concatenate(data[test_size + dev_size:])

    print('Train:', train.shape)
    print('Dev:', dev.shape)
    print('Test:', test.shape)

    np.savetxt('./data_waveform_1_split/train.csv', train, delimiter=',')
    np.savetxt('./data_waveform_1_split/dev.csv', dev, delimiter=',')
    np.savetxt('./data_waveform_1_split/test.csv', test, delimiter=',')

    print("--- %s seconds ---" % (time.time() - start_time))

parse_raw()
# data_split()