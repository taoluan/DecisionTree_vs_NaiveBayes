from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
import numpy as np

iris = load_iris()
X = iris.data
y = iris.target
#print(X)
#print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)

model = GaussianNB()
model.fit(X_train,y_train)
#print(model)

thucte = y_test
dubao = model.predict(X_test)
#print(thucte)
#print(dubao)

cnf_matrix_gnb = confusion_matrix(thucte,dubao)
#print(cnf_matrix_gnb)

kf= KFold(n_splits=15)
for train_index, test_index in kf.split(X):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index,], X[test_index,]
    y_train, y_test = y[train_index], y[test_index]
    #print("X_test", X_test)

model = GaussianNB()
model.fit(X_train,y_train)
print(model)

thucte = y_test
dubao = model.predict(X_test)
print(thucte)
print(dubao)

cnf_matrix_gnb = confusion_matrix(thucte,dubao)
print(cnf_matrix_gnb)