import numpy as np
import json


data = {
    'rnn': {
        'dev': {
            'pred': list(np.loadtxt('./rnn_results/dev.pred.txt')),
            'target': list(np.loadtxt('./rnn_results/dev.target.txt')),
        },
        'test': {
            'pred': list(np.loadtxt('./rnn_results/test.pred.txt')),
            'target': list(np.loadtxt('./rnn_results/test.target.txt')),
        },
    },
    'transformer': {
        'dev': {
            'pred': list(np.loadtxt('./transformer_results/dev.pred.txt')),
            'target': list(np.loadtxt('./transformer_results/dev.target.txt')),
        },
        'test': {
            'pred': list(np.loadtxt('./transformer_results/test.pred.txt')),
            'target': list(np.loadtxt('./transformer_results/test.target.txt')),
        },
    },
}
with open('./visualization/public/data.json', 'w') as fp:
    json.dump(data, fp)

