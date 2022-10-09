import csv
import math
import random
import time
import os
from pathlib import Path

import numpy as np
import torch
import torchaudio

seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

start_time = time.time()

TXT_PATH = './raw/LC rhythm - all LC Tuanan/Raven tables'
WAV_PATH = './raw/LC rhythm - all LC Tuanan'

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

parse_raw()