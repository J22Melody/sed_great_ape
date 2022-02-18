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
torch.manual_seed(0)
np.random.seed(0)

start_time = time.time()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_PATH = '../data'
TXT_PATH = '{}/data_full/Klaus_Zuberbuhler_1994_2002/Raven Pro Tables'.format(DATA_PATH)
WAV_PATH = '{}/data_full/Klaus_Zuberbuhler_1994_2002/Recordings'.format(DATA_PATH)

def parse_raw():
    bundle = torchaudio.pipelines.WAV2VEC2_BASE
    model = bundle.get_model().to(device)

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
        waveform = waveform.to(device)

        # transform
        new_sample_rate = bundle.sample_rate
        transformed = torchaudio.functional.resample(waveform, sample_rate, new_sample_rate)

        # reshape
        length = transformed.size()[1]
        new_length = math.ceil(length / new_sample_rate) * new_sample_rate
        target = torch.zeros(1, new_length).to(device)
        target[:, :length] = transformed
        target = target.view(-1, new_sample_rate)

        # wav2vec2
        with torch.inference_mode():
            features, _ = model.extract_features(target)
        output_features = features[len(features) - 1]
        output_features = output_features.view(output_features.size()[0], -1).cpu()

        # read txt annotations
        calls = None
        with open(txtpath, newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            data = list(reader)
            calls = [(float(item[3]), float(item[4])) for item in data[1:]]

        # create y
        y = []
        for i in range(output_features.size()[0]):
            is_call = 0
            t_begin = i
            t_end = i + 1
            for begin, end in calls:
                if not (t_begin > end or t_end < begin):
                    is_call = 1
                    break
            y.append(is_call)
        y = torch.tensor(y).view(-1, 1)

        data = torch.cat((y, output_features), dim=1)

        np.savetxt('{}/data_wav2vec2_1/{}.csv'.format(DATA_PATH, filename), data.numpy(), delimiter=',')

def data_split():
    paths = Path('{}/data_wav2vec2_1'.format(DATA_PATH)).rglob('*.csv')
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

    np.savetxt('{}/data_wav2vec2_1_split/train.csv'.format(DATA_PATH), train, delimiter=',')
    np.savetxt('{}/data_wav2vec2_1_split/dev.csv'.format(DATA_PATH), dev, delimiter=',')
    np.savetxt('{}/data_wav2vec2_1_split/test.csv'.format(DATA_PATH), test, delimiter=',')

    print("--- %s seconds ---" % (time.time() - start_time))

parse_raw()
data_split()