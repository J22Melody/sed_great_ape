import re
from statistics import mean, stdev
from pprint import pprint
import pandas as pd

MODELS = [
    'chimp_waveform_lstm',
    'chimp_spectrogram_lstm',
    'chimp_wav2vec2_lstm',
    'chimp_wav2vec2_transformer',
    'chimp_wav2vec2_lstm_batch_4',
    'chimp_wav2vec2_lstm_batch_8',
    'chimp_wav2vec2_lstm_dropout_0.2',
    'chimp_wav2vec2_lstm_dropout_0.1',
    'chimp_wav2vec2_lstm_bw',
    'chimp_wav2vec2_lstm_ar',
    # 'longcall_wav2vec2_lstm_ar',
    'longcall_wav2vec2_lstm_binary_ar',
    # 'bonobo_wav2vec2_lstm_ar',
    'bonobo_wav2vec2_lstm_binary_ar',
]

SEEDS = [0, 42, 3407]

stats_all = {}

for model in MODELS:
    stats = {
        'dev_accuracy': [],
        'dev_f1_avg': [],
        'test_accuracy': [],
        'test_f1_avg': [],
        'dev_f1': [],
        'test_f1': [],
        'dev_precision': [],
        'dev_recall': [],
        'test_precision': [],
        'test_recall': [],
    }

    for seed in SEEDS:
        log_file_path = './models/{}_{}/train.log'.format(model, seed)
        f = open(log_file_path, "r")
        lines = f.read().splitlines()
        data_set = 'test'

        print('\n'.join(lines))

        for i, line in enumerate(lines):
            if line.startswith('Test Epoch:'):
                text = []
                for j in range(100):
                    if lines[i+j].startswith('---'):
                        break
                    else:
                        text.append(lines[i+j])
                text = ' '.join(text)

                stats[data_set + '_accuracy'].append(float(re.search('accuracy: (\d+\.\d+)', text).group(1)))
                stats[data_set +'_f1_avg'].append(float(re.search('f1_avg: (\d+\.\d+)', text).group(1)))
                stats[data_set +'_precision'].append([float(n) for n in re.search('precision: \[(.*?)\]', text).group(1).split()])
                stats[data_set +'_recall'].append([float(n) for n in re.search('recall: \[(.*?)\]', text).group(1).split()])
                stats[data_set +'_f1'].append([float(n) for n in re.search('f1: \[(.*?)\]', text).group(1).split()])

                data_set = 'dev'

    # pprint(stats, sort_dicts=False)

    for key in stats:
        if isinstance(stats[key][0], float):
            stats[key] = '{:.1f}±{:.1f}'.format(mean(stats[key]) * 100, stdev(stats[key]) * 100)
        else:
            stats[key]  = ['{:.1f}±{:.1f}'.format(mean(l) * 100, stdev(l) * 100) for l in list(zip(*stats[key]))]

    # pprint(stats, sort_dicts=False)

    stats_all[model] = stats

# pprint(stats_all)

df = pd.DataFrame.from_dict(stats_all, orient='index')

df.to_csv('./model_stats.csv')