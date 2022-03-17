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

TXT_PATH = './data_greatarc/wetransfer_long-call-recordings-raven-acoustics_2021-11-22_1051/LC rhythm - all LC Tuanan/Raven tables'
WAV_PATH = './data_greatarc/wetransfer_long-call-recordings-raven-acoustics_2021-11-22_1051/LC rhythm - all LC Tuanan'

def parse_raw():
    pulse_level_map = {}

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
            reader = csv.DictReader(f, delimiter='\t')
            calls = []
            for item in reader:
                pulse_level = item['Pulse level'].strip() if item.get('Pulse level', '') else 'Unknown'

                # fix typo
                if pulse_level in ['Sub-pulse trnasitory element', 'Sub-pule transitory element', 'Sub-pulse transiotry element']:
                    pulse_level = 'Sub-pulse transitory element'

                if pulse_level in pulse_level_map:
                    pulse_level_map[pulse_level]['count'] += 1
                else:
                    pulse_level_map[pulse_level] = {
                        'count': 1,
                        'id': len(pulse_level_map.keys()) + 1,
                    }
                call = [float(item['Begin Time (s)']), float(item['End Time (s)']), pulse_level_map[pulse_level]['id']]
                calls.append(call)
        # print('calls:', calls)

        # create training data
        def get_y(t):
            y = 0
            t_begin = t - T_real / 2
            t_end = t + T_real / 2
            for begin, end, call_type in calls:
                if not (t_begin > end or t_end < begin):
                    y = call_type
                    break
            return y
        vfunc = np.vectorize(get_y)
        y = vfunc(times)
        y = np.expand_dims(y, axis=1)
        # print('y:', y.shape)

        spectrogram = np.swapaxes(spectrogram, 0, 1)
        data = np.concatenate((y, spectrogram), axis=1)
        print('data:')
        print(data.shape)

        np.savetxt('./data_longcall_1/{}.csv'.format(filename), data, delimiter=',')
        print(pulse_level_map)

    print("--- %s seconds ---" % (time.time() - start_time))

def data_split():
    paths = list(Path('./data_longcall_1').rglob('*.csv'))
    print('Split {} files: {}'.format(len(paths), paths))

    # data = [np.loadtxt(path, delimiter=',') for path in paths]
    # random.shuffle(data)

    data = np.concatenate([np.loadtxt(path, delimiter=',') for path in paths])

    # data = data[data[:, 0] != 0]

    unique, counts = np.unique(data[:, 0], return_counts=True)
    counts_by_id = dict(zip(unique, counts))
    print('Labels in data: ', counts_by_id)

    train, test_dev = train_test_split(data, test_size=0.2, random_state=seed)
    dev, test = train_test_split(test_dev, test_size=0.5, random_state=seed)

    # upsampling
    # upsample_classes = [-1, 1, 2, 3]
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

    np.savetxt('./data_longcall_1_split/train.csv', train, delimiter=',')
    np.savetxt('./data_longcall_1_split/dev.csv', dev, delimiter=',')
    np.savetxt('./data_longcall_1_split/test.csv', test, delimiter=',')

    print("--- %s seconds ---" % (time.time() - start_time))

def data_split_by_file():
    paths = list(Path('./data_longcall_1').rglob('*.csv'))
    print('Split {} files: {}'.format(len(paths), paths))

    data = [np.loadtxt(path, delimiter=',') for path in paths]
    random.shuffle(data)

    train_index = int(len(data) * 0.8)
    dev_index = int(len(data) * 0.9)
    train = np.concatenate(data[:train_index])
    dev = np.concatenate(data[train_index:dev_index])
    test = np.concatenate(data[dev_index:])

    np.random.shuffle(train)

    print('Train:', train.shape)
    print('Dev:', dev.shape)
    print('Test:', test.shape)

    np.savetxt('./data_longcall_1_split_by_file/train.csv', train, delimiter=',')
    np.savetxt('./data_longcall_1_split_by_file/dev.csv', dev, delimiter=',')
    np.savetxt('./data_longcall_1_split_by_file/test.csv', test, delimiter=',')

    print("--- %s seconds ---" % (time.time() - start_time))

# parse_raw()
# data_split()
data_split_by_file()
