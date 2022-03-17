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
import itertools


start_time = time.time()

seed = 42
random.seed(seed)
np.random.seed(seed)

train = np.loadtxt('./data_longcall_1_split_by_file/train.csv', delimiter=',')
test = np.concatenate([np.loadtxt('./data_longcall_1_split_by_file/test.csv', delimiter=','), np.loadtxt('./data_longcall_1_split/dev.csv', delimiter=',')])

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
    MLPClassifier(hidden_layer_sizes=(100), alpha=100, learning_rate_init=0.0001, random_state=30, early_stopping=True, n_iter_no_change=15, verbose=True),
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

print("--- %s seconds ---" % (time.time() - start_time))