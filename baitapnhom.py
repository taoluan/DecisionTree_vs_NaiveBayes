import pandas as pd
import numpy as np
import random as rd
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

###Cây quyết định-đánh giá bằng hold-out
dt=pd.read_csv("magic04.data",delimiter=",")
x = dt.iloc[0:,0:10]
y = dt.iloc[:,10:11]
#print(len(x))
#plt.scatter(x,y)
#plt.show()
#print(x)
#print(np.unique(y))
#cay quyet dinh - holdout
def cayquyetdinh_holdout_gini(x,y):
    X_train,X_test,y_train,y_test = train_test_split(x,y, test_size=1/3.0, random_state=100)
    nghithuc_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=8, min_samples_leaf=1)
    nghithuc_gini.fit(X_train, y_train)
    #test = dt.iloc[14516:14517,0:10]
    #print(test)
    #y_pred = nghithuc_gini.predict([[28.2452,7.4431,2.4609,0.5848,0.3581,11.1731,10.0138,-4.6976,76.566,83.9864]])
    #print(y_pred)
    y_pred = nghithuc_gini.predict(X_test)
    #print(y_pred)
    print("Độ chính xác cây quyết định khi sử dụng nghi thức phân chia dl Hold-out",accuracy_score(y_test,y_pred)*100)
    #print(confusion_matrix(y_test, y_pred))
def cayquyetdinh_holdout_entropy(x,y):
    X_train,X_test,y_train,y_test = train_test_split(x,y, test_size=1/3.0, random_state=100)
    nghithuc_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,max_depth=8, min_samples_leaf=1)
    nghithuc_entropy.fit(X_train, y_train)
    y_pred = nghithuc_entropy.predict(X_test)
    print("Độ chính xác cây quyết định khi sử dụng nghi thức phân chia dl Hold-out",accuracy_score(y_test,y_pred)*100)
###Cây Bayes thơ ngây - KFold    
def cayquyetdinh_vs_caybayes_kfold(X,Y):
    accuracy_bayes = 0
    accuracy_cqd = 0
    model = GaussianNB()
    kf= KFold(n_splits=5,shuffle=True,random_state=True)
    for train_index, test_index in kf.split(X):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X.iloc[train_index,], X.iloc[test_index,]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        nghithuc_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=8, min_samples_leaf=1)
        nghithuc_gini.fit(X_train, y_train)
        y_pred = nghithuc_gini.predict(X_test)
        accuracy_cqd += accuracy_score(y_test,y_pred)
        print("cay quyet dinh do chinh xac",accuracy_score(y_test,y_pred)*100)
        model.fit(X_train, y_train.values.ravel())
        thucte = y_test
        dubao = model.predict(X_test)
        accuracy_bayes += accuracy_score(thucte,dubao)
        print("cay bayes do chinh xac",accuracy_score(thucte,dubao)*100)
        
    #print("Accuracy Bayes-kfold = ",(accuracy_bayes/5)*100)
    #print("Accuracy Cayquyetdinh-kfold = ",(accuracy_cqd/5)*100)
def cayquyetdinh_vs_caybayes_holdout(x,y):
    model = GaussianNB()
    accuracy_bayes = 0
    accuracy_cqd = 0
    for i in range(0,3):
        X_train,X_test,y_train,y_test = train_test_split(x,y, test_size=1/3.0, random_state=100+i*10,shuffle=True)
        nghithuc_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=9, min_samples_leaf=1)
        nghithuc_gini.fit(X_train, y_train)
        y_pred = nghithuc_gini.predict(X_test)
        accuracy_cqd += accuracy_score(y_test,y_pred)
        print("cay quyet dinh Lan lap",i," do chinh xac",accuracy_score(y_test,y_pred)*100)
        model.fit(X_train, y_train.values.ravel())
        thucte = y_test
        dubao = model.predict(X_test)
        accuracy_bayes += accuracy_score(thucte,dubao)
        print("Bayes tho ngay Lan lap",i," do chinh xac",accuracy_score(thucte,dubao)*100) 
    print("Accuracy Bayes-holdout = ",(accuracy_bayes/5)*100)
    print("Accuracy Cayquyetdinh-holdout = ",(accuracy_cqd/5)*100)
#cayquyetdinh_holdout_gini(x,y)
#cayquyetdinh_holdout_entropy(x,y)
#caybayes_kfold(x,y)
#2-3 thuoc tinh 7-10 du lieu xay dung mo hinh
cayquyetdinh_vs_caybayes_holdout(x,y)
#cayquyetdinh_vs_caybayes_kfold(x,y)