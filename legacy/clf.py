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

# number_all = 100
# number_test = 20

# # paths = Path('./data').rglob('*.csv')
# paths = Path('./data_full_clf_1').rglob('*.csv')
# paths = list(itertools.islice(paths, number_all))
# print('Split {} files: {}'.format(len(paths), paths))
# data = [np.loadtxt(path, delimiter=',') for path in paths]
# random.shuffle(data)

# train = np.concatenate((data[:-number_test]), axis=0)
# test = np.concatenate((data[-number_test:]), axis=0)
# # data = np.concatenate(data, axis=0)
# # train, test = train_test_split(data, test_size=0.1, random_state=42)

train = np.loadtxt('./data_full_clf_1_split/train.csv', delimiter=',')
test = np.loadtxt('./data_full_clf_1_split/test.csv', delimiter=',')

print('train:', train.shape)
print('test:', test.shape)

X_train = train[:, 1:]
y_train = train[:, 0]
X_test = test[:, 1:]
y_test = test[:, 0]

pipe = make_pipeline(
    preprocessing.StandardScaler(), 
    # LogisticRegression(C=1),
    MLPClassifier(hidden_layer_sizes=(100), alpha=1, learning_rate_init=0.001, random_state=30, verbose=True),
)
pipe.fit(X_train, y_train)

print('-----training-----')
y_predict = pipe.predict(X_train)
print('precision_score: ', precision_score(y_train, y_predict))
print('recall_score: ', recall_score(y_train, y_predict))
print('f1_score: ', f1_score(y_train, y_predict))
print('accuracy_score: ', accuracy_score(y_train, y_predict))

display = PrecisionRecallDisplay.from_estimator(
    pipe, X_train, y_train, name="Model"
)
_ = display.ax_.set_title("2-class Precision-Recall curve")
plt.show()

print('-----testing-----')
y_predict = pipe.predict(X_test)
print('precision_score: ', precision_score(y_test, y_predict))
print('recall_score: ', recall_score(y_test, y_predict))
print('f1_score: ', f1_score(y_test, y_predict))
print('accuracy_score: ', accuracy_score(y_test, y_predict))

display = PrecisionRecallDisplay.from_estimator(
    pipe, X_test, y_test, name="Model"
)
_ = display.ax_.set_title("2-class Precision-Recall curve")
plt.show()

print("--- %s seconds ---" % (time.time() - start_time))