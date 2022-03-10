import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, PrecisionRecallDisplay
import matplotlib.pyplot as plt
import time
import random
from pathlib import Path


start_time = time.time()

seed = 42
random.seed(seed)
np.random.seed(seed)

# train_paths = [
#     'Kelly experiment spots 11-1-2011',
# ]
# test_paths = [
#     'Kelly experiment Tiger 19-12-2010',
#     # 'YetYeni experiment tiger sheet 10-02-2011 (1)_1st half',
# ]
# train = np.concatenate([np.loadtxt('./data_greatarc_1/{}.csv'.format(path), delimiter=',') for path in train_paths])
# test = np.concatenate([np.loadtxt('./data_greatarc_1/{}.csv'.format(path), delimiter=',') for path in test_paths])

# random.shuffle(train)

# train = train[train[:, 0] != 0]
# test = test[test[:, 0] != 0]

train = np.loadtxt('./data_greatarc_1_split_nonzero/train.csv', delimiter=',')
test = np.loadtxt('./data_greatarc_1_split_nonzero/test.csv', delimiter=',')

print('train:', train.shape)
print('test:', test.shape)

X_train = train[:, 1:]
y_train = train[:, 0]
X_test = test[:, 1:]
y_test = test[:, 0]

pos_y_train = np.count_nonzero(y_train)
print('Positive labels in train: ', pos_y_train)
pos_y_test = np.count_nonzero(y_test)
print('Positive labels in test: ', pos_y_test)

unique, counts = np.unique(y_train, return_counts=True)
print('Labels in train: ', dict(zip(unique, counts)))
unique, counts = np.unique(y_test, return_counts=True)
print('Labels in test: ', dict(zip(unique, counts)))

pipe = make_pipeline(
    preprocessing.StandardScaler(),
    # LogisticRegression(C=1),
    MLPClassifier(hidden_layer_sizes=(50), alpha=100, learning_rate_init=0.0001, random_state=30, early_stopping=True, verbose=True),
)
pipe.fit(X_train, y_train)

print('-----training-----')
y_predict = pipe.predict(X_train)
print('precision_score: ', precision_score(y_train, y_predict, average=None, zero_division=1))
print('recall_score: ', recall_score(y_train, y_predict, average=None))
print('f1_score: ', f1_score(y_train, y_predict, average=None))
print('accuracy_score: ', accuracy_score(y_train, y_predict))

# display = PrecisionRecallDisplay.from_estimator(
#     pipe, X_train, y_train, name="Model"
# )
# _ = display.ax_.set_title("2-class Precision-Recall curve")
# plt.show()

print('-----testing-----')
y_predict = pipe.predict(X_test)
print('precision_score: ', precision_score(y_test, y_predict, average=None, zero_division=1))
print('recall_score: ', recall_score(y_test, y_predict, average=None))
print('f1_score: ', f1_score(y_test, y_predict, average=None))
print('accuracy_score: ', accuracy_score(y_test, y_predict))

# display = PrecisionRecallDisplay.from_estimator(
#     pipe, X_test, y_test, name="Model"
# )
# _ = display.ax_.set_title("2-class Precision-Recall curve")
# plt.show()

# Plot Precision-Recall curve for each class and iso-f1 curves
# colors = ["navy", "turquoise", "darkorange", "cornflowerblue", "teal"]

# _, ax = plt.subplots(figsize=(7, 8))

# f_scores = np.linspace(0.2, 0.8, num=4)
# lines, labels = [], []
# for f_score in f_scores:
#     x = np.linspace(0.01, 1)
#     y = f_score * x / (2 * x - f_score)
#     (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
#     plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

# n_classes = 3
# for i, color in zip(range(n_classes), colors):
#     display = PrecisionRecallDisplay(
#         recall=recall[i+1],
#         precision=precision[i+1],
#     )
#     display.plot(ax=ax, name=f"Precision-recall for class {i+1}", color=color)

# # add the legend for the iso-f1 curves
# handles, labels = display.ax_.get_legend_handles_labels()
# handles.extend([l])
# labels.extend(["iso-f1 curves"])
# # set the legend and the axes
# ax.set_xlim([0.0, 1.0])
# ax.set_ylim([0.0, 1.05])
# ax.legend(handles=handles, labels=labels, loc="best")
# ax.set_title("Extension of Precision-Recall curve to multi-class")

# plt.show()

print("--- %s seconds ---" % (time.time() - start_time))