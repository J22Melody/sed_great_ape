from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt


wavpath = './data/chunk-13.wav'

print('processing {} ...'.format(wavpath))

# read wav file
# https://stackoverflow.com/questions/44787437/how-to-convert-a-wav-file-to-a-spectrogram-in-python3
sample_rate, samples = wavfile.read(wavpath)
print('sample rate:', sample_rate)
print('number of samples:', len(samples))
print('duration (sec):', len(samples) / sample_rate)

T_real = 0.1
T = T_real * 8 / 7
nperseg = int(sample_rate * T)
noverlap = nperseg // 8

# frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate, nperseg=nperseg, noverlap=noverlap)
frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
print('frequencies:')
print(frequencies, len(frequencies))
print('times:')
print(times, len(times))
print('spectrogram:')
print(spectrogram, spectrogram.shape)

# plt.pcolormesh(times, frequencies, spectrogram)
plt.pcolormesh(times, frequencies[:700], spectrogram[:700])
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
