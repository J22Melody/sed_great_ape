import csv
import numpy as np
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import time
import itertools
from pathlib import Path
import os
import random
import math

from sklearn.utils import resample
from sklearn.model_selection import train_test_split


seed = 42
random.seed(seed)
np.random.seed(seed)

start_time = time.time()

TXT_PATH = './data_greatarc/wetransfer_audio-files-for-zifan_2021-12-14_1301'
WAV_PATH = './data_greatarc/wetransfer_audio-files-for-zifan_2021-12-14_1301'

def parse_raw():
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
        # https://stackoverflow.com/questions/44787437/how-to-convert-a-wav-file-to-a-spectrogram-in-python3
        sample_rate, samples = wavfile.read(wavpath)
        # print('sample rate:', sample_rate)
        # print('number of samples:', len(samples))
        # print('duration (sec):', len(samples) / sample_rate)

        T_real = 1
        T = T_real * 8 / 7
        nperseg = int(sample_rate * T)
        noverlap = nperseg // 8

        frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate, nperseg=nperseg, noverlap=noverlap)
        # frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
        # print('frequencies:')
        # print(frequencies, len(frequencies))
        # print('times:')
        # print(times, len(times))
        # print('spectrogram:')
        # print(spectrogram, spectrogram.shape)

        # plt.pcolormesh(times, frequencies, spectrogram)
        # plt.ylabel('Frequency [Hz]')
        # plt.xlabel('Time [sec]')
        # plt.show()

        # read txt annotations
        calls = None
        with open(txtpath, newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            data = list(reader)
            calls = [(float(item[3]), float(item[4])) for item in data[1:]]
        # print('calls:', calls)

        # create training data
        def get_y(t):
            is_call = 0
            t_begin = t - T_real / 2
            t_end = t + T_real / 2
            for begin, end in calls:
                if not (t_begin > end or t_end < begin):
                    is_call = 1
                    break
            return is_call
        vfunc = np.vectorize(get_y)
        y = vfunc(times)
        y = np.expand_dims(y, axis=1)
        # print('y:', y.shape)

        spectrogram = np.swapaxes(spectrogram, 0, 1)
        data = np.concatenate((y, spectrogram), axis=1)
        # print('data:')
        # print(data, data.shape)

        np.savetxt('./data_greatarc_1/{}.csv'.format(filename), data, delimiter=',')

    print("--- %s seconds ---" % (time.time() - start_time))

def data_split():
    train_paths = [
        'ElisaEtek experiment Blue sheet 6-1-2011',
        'Kelly experiment spots 11-1-2011',
        'Kelly experiment Tiger 19-12-2010',
    ]
    test_dev_paths = [
        'YetYeni experiment tiger sheet 10-02-2011 (1)_1st half',
        'YetYeni experiment tiger sheet 10-02-2011 (3)',
    ]

    train = np.concatenate([np.loadtxt('./data_greatarc_1/{}.csv'.format(path), delimiter=',') for path in train_paths])
    test_dev = np.concatenate([np.loadtxt('./data_greatarc_1/{}.csv'.format(path), delimiter=',') for path in test_dev_paths])
    dev, test = train_test_split(test_dev, test_size=0.5, random_state=seed)

    random.shuffle(train)

    # downsampling
    # train_1 = train[train[:, 0] == 1]
    # train_0 = train[train[:, 0] == 0]
    # train_0 = resample(train_0, random_state=seed, n_samples=train_1.shape[0], replace=False)
    # train = np.concatenate([train_0, train_1])
    # np.random.shuffle(train)

    print('Train:', train.shape)
    print('Dev:', dev.shape)
    print('Test:', test.shape)

    np.savetxt('./data_greatarc_1_split/train.csv', train, delimiter=',')
    np.savetxt('./data_greatarc_1_split/dev.csv', dev, delimiter=',')
    np.savetxt('./data_greatarc_1_split/test.csv', test, delimiter=',')

    print("--- %s seconds ---" % (time.time() - start_time))

# parse_raw()
data_split()
