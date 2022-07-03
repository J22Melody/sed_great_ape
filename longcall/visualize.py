import numpy as np
import matplotlib.pyplot as plt


pred = np.loadtxt('./rnn.pred.txt')
target = np.loadtxt('./rnn.target.txt')

import json
data = {
    'pred': list(pred),
    'target': list(target),
}
with open('./visualization/public/data.json', 'w') as fp:
    json.dump(data, fp)


# assert pred.shape == target.shape
# length = pred.shape[0]

# x = np.arange(0, length * 20, 20)

# plt.plot(x, target, label = "line 1")
# plt.plot(x, pred, label = "line 2")
# plt.legend()
# plt.show()