import csv
import numpy as np
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import time
from pathlib import Path
import os

start_time = time.time()

TXT_PATH = './data_full/Klaus_Zuberbuhler_1994_2002/Raven Pro Tables'
WAV_PATH = './data_full/Klaus_Zuberbuhler_1994_2002/Recordings'

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

    np.savetxt('./data_full_clf_1/{}.csv'.format(filename), data, delimiter=',')


print("--- %s seconds ---" % (time.time() - start_time))