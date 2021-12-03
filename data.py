import csv
import numpy as np
from scipy import signal
from scipy.io import wavfile

filenames = ['96-22b124-L', '96-41b228-L', '96-44b243-E2']

for filename in filenames:
    print('processing {} ...'.format(filename))

    # read wav file
    # https://stackoverflow.com/questions/44787437/how-to-convert-a-wav-file-to-a-spectrogram-in-python3
    sample_rate, samples = wavfile.read('./data/{}.wav'.format(filename))
    print('sample rate:', sample_rate)
    print('number of samples:', len(samples))
    print('duration (sec):', len(samples) / sample_rate)

    T_real = 0.2
    T = T_real * 8 / 7
    nperseg = int(sample_rate * T)
    noverlap = nperseg // 8

    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate, nperseg=nperseg, noverlap=noverlap)
    print('frequencies:')
    print(frequencies, len(frequencies))
    print('times:')
    print(times, len(times))
    print('spectrogram:')
    print(spectrogram, spectrogram.shape)

    # read txt annotations
    calls = None
    with open('./data/{}.txt'.format(filename), newline='') as f:
        reader = csv.reader(f, delimiter='\t')
        data = list(reader)
        calls = [(float(item[3]), float(item[4])) for item in data[1:]]
    print('calls:', calls)

    # create training data
    def get_y(t):
        is_call = 0
        for begin, end in calls:
            if t > begin and t < end:
                is_call = 1
                break
        return is_call
    vfunc = np.vectorize(get_y)
    y = vfunc(times)
    y = np.expand_dims(y, axis=1)
    print('y:', y.shape)

    spectrogram = np.swapaxes(spectrogram, 0, 1)
    data = np.concatenate((y, spectrogram), axis=1)
    print('data:')
    print(data, data.shape)

    np.savetxt('./data/{}.csv'.format(filename), data, delimiter=',')


