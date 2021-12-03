import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = [
    np.loadtxt('./data/96-22b124-L.csv', delimiter=','),
    np.loadtxt('./data/96-41b228-L.csv', delimiter=','),
    np.loadtxt('./data/96-44b243-E2.csv', delimiter=','),
]

train = np.concatenate((data[0], data[1]), axis=0)
test = data[2]
print('train:', train.shape)
print('test:', test.shape)

X_train = train[:, 1:]
y_train = train[:, 0]
X_test = test[:, 1:]
y_test = test[:, 0]

clf = LogisticRegression(random_state=30)
clf.fit(X_train, y_train)

print('-----training-----')
y_predict = clf.predict(X_train)
print('accuracy_score: ', accuracy_score(y_train, y_predict))

print('-----testing-----')
y_predict = clf.predict(X_test)
print('accuracy_score: ', accuracy_score(y_test, y_predict))
