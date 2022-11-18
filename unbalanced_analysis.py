from pathlib import Path
import numpy as np

MODEL_PATH = './models/longcall_wav2vec2_lstm_binary_ar_0_bonobo'
RESULTS_PATH = '{}/results/'.format(MODEL_PATH)

dist_list = []
for path in Path(RESULTS_PATH).rglob('*.dist.txt'):
    dist = np.loadtxt(path, delimiter=',')
    dist_list.append(dist)

target_list = []
for path in Path(RESULTS_PATH).rglob('*.target.txt'):
    target = np.loadtxt(path, delimiter=',')
    target_list.append(target)

dist = np.concatenate(dist_list)
target = np.concatenate(target_list)

print(dist.shape)
print(target.shape)