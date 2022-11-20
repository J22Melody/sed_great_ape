from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay


MODEL_PATHS = [
    './models/bonobo_wav2vec2_lstm_binary_ar_0',
    './models/longcall_wav2vec2_lstm_binary_ar_0',
    './models/longcall_wav2vec2_lstm_binary_ar_0_bonobo',
]

for MODEL_PATH in MODEL_PATHS:
    RESULTS_PATH = '{}/results/'.format(MODEL_PATH)
    PIC_PATH = '{}/unbalanced_analysis.png'.format(MODEL_PATH)

    dist_list = []
    target_list = []
    for path in Path(RESULTS_PATH).rglob('*.dist.txt'):
        dist = np.loadtxt(path, delimiter=',')
        dist_list.append(dist)
        target = np.loadtxt(str(path).replace('.dist', '.target'), delimiter=',')
        target_list.append(target)

    dist = np.concatenate(dist_list)[:, 1]
    target = np.concatenate(target_list)

    display = PrecisionRecallDisplay.from_predictions(target, dist)

    plt.savefig(PIC_PATH)