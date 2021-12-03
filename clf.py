import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, PrecisionRecallDisplay
import matplotlib.pyplot as plt

data = [
    np.loadtxt('./data/96-22b124-L.csv', delimiter=','),
    np.loadtxt('./data/96-41b228-L.csv', delimiter=','),
    np.loadtxt('./data/96-44b243-E2.csv', delimiter=','),
]

# train = np.concatenate((data[0], data[2]), axis=0)
# test = data[1]
data = np.concatenate((data[0], data[1], data[2]), axis=0)
train, test = train_test_split(data, test_size=0.1, random_state=42)

print('train:', train.shape)
print('test:', test.shape)

X_train = train[:, 1:]
y_train = train[:, 0]
X_test = test[:, 1:]
y_test = test[:, 0]

pipe = make_pipeline(
    preprocessing.StandardScaler(), 
    # LogisticRegression(class_weight='balanced', C=1),
    MLPClassifier(hidden_layer_sizes=(50), alpha=1, random_state=30, early_stopping=True),
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