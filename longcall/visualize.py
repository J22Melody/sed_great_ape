import numpy as np
import json


dev_pred = np.loadtxt('./rnn_results/dev.pred.txt')
dev_target = np.loadtxt('./rnn_results/dev.target.txt')
test_pred = np.loadtxt('./rnn_results/test.pred.txt')
test_target = np.loadtxt('./rnn_results/test.target.txt')


data = {
    'dev': {
        'pred': list(dev_pred),
        'target': list(dev_target),
    },
    'test': {
        'pred': list(test_pred),
        'target': list(test_target),
    },
}
with open('./visualization/public/data.json', 'w') as fp:
    json.dump(data, fp)

