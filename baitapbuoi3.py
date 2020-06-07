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
from sklearn.tree import DecisionTreeRegressor 
from sklearn.tree import DecisionTreeClassifier
###1
wineWhite=pd.read_csv("winequality-white.csv",delimiter=";")
thuoctinh = wineWhite.iloc[0,0:11]
nhan = wineWhite.iloc[:,11:12]
X = wineWhite.iloc[:,0:11]
y = wineWhite.iloc[:,11:12]
#print(X)
#print(y)
#print (len(thuoctinh)) 
#>> Dư liệu có 11 thuộc tính
#print(y) 
#>> Nhan la cot Quality
#print(np.unique(y)) 
#>> Gía trị nhãn:  [3 4 5 6 7 8 9]
###2
model = GaussianNB()
temp = 0
kf= KFold(n_splits=70,shuffle=True,random_state=True)
for train_index, test_index in kf.split(X):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X.iloc[train_index,], X.iloc[test_index,]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    ###4 Đánh giá độ chính xác cho mõi lần lập
    model.fit(X_train, y_train.values.ravel())
    #print(model)
    thucte = y_test
    dubao = model.predict(X_test)
    temp += accuracy_score(thucte,dubao)
    #print(thucte)
    #print(dubao)
    ###4 Đánh giá độ chính xác
    #print("Độ chính xác mõi lần lập ",accuracy_score(thucte,dubao)*100)
    #cnf_matrix_gnb = confusion_matrix(thucte,dubao)
    #print(cnf_matrix_gnb)
    #print("X_test", X_test)
print(len(X_test)) 
#>> số lượng phần tử trong tập test là 69
print(len(X_train)) 
#>> số lượng phần tử trong tập huấn luyện la 4829
###3: Dự đoán nhãn
#model.fit(X_train, y_train.values.ravel())
#print(model)
#thucte = y_test
#dubao = model.predict(X_test)
print("Độ chính xác cây Bayes thơ ngây khi sử dụng nghi thức phân chia dl KFold",(temp/70)*100)
#print(thucte)
#print(dubao)
###4 Đánh giá độ chính xác o vong lap cuoi la : Độ chính xác mõi lần lập  46.3768115942029
###5 Đánh giá độ chính xác cho 70 lần lập
#print("Độ chính xác 70 lần lập",accuracy_score(thucte,dubao)*100)
#Độ chính xác 70 lần lập 46.3768115942029
cnf_matrix_gnb = confusion_matrix(thucte,dubao)
#print(cnf_matrix_gnb)
#[[ 0  1  1  0  0]
 #[ 0 17 12  2  0]
 #[ 0  6  9 12  0]
 #[ 0  0  1  6  0]
 #[ 0  0  0  2  0]]
#Độ chính xác 70 lần lập 46.3768115942029
#Độ chính xác mõi lần lập  37.142857142857146
#Độ chính xác mõi lần lập  40.0
#Độ chính xác mõi lần lập  52.85714285714286
#Độ chính xác mõi lần lập  52.85714285714286
#Độ chính xác mõi lần lập  41.42857142857143
#Độ chính xác mõi lần lập  41.42857142857143
#...........................................
#Độ chính xác mõi lần lập  45.714285714285715
#Độ chính xác mõi lần lập  37.142857142857146
#Độ chính xác mõi lần lập  30.0
#Độ chính xác mõi lần lập  39.130434782608695
#Độ chính xác mõi lần lập  46.3768115942029
###6 cây quyết định 
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3, random_state=0)
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 0,
                               max_depth=3, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)
y_pred = clf_gini.predict(X_test)
print("Độ chính xác khi sử dụng gt Cây quyết định và Hold-out",accuracy_score(y_test,y_pred)*100)
#Độ chính xác 51.02040816326531
#
kf= KFold(n_splits=70,shuffle=True,random_state=True)
for train_index, test_index in kf.split(X):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X.iloc[train_index,], X.iloc[test_index,]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 0,
                               max_depth=3, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)
y_pred = clf_gini.predict(X_test)
print("Độ chính xác khi sử dụng gt Cây quyết định và KFol",accuracy_score(y_test,y_pred)*100)
#Độ chính xác 57.971014492753625
#Cay Bayes thơ ngây
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3, random_state=0)
model.fit(X_train, y_train.values.ravel())
#print(model)
thucte = y_test
dubao = model.predict(X_test)
print("Độ chính xác khi sử dụng gt Cây Bayes và Hold-out",accuracy_score(thucte,dubao)*100)
#Độ chính xác 44.89795918367347
kf= KFold(n_splits=70,shuffle=True,random_state=True)
for train_index, test_index in kf.split(X):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X.iloc[train_index,], X.iloc[test_index,]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
model.fit(X_train, y_train.values.ravel())
thucte = y_test
dubao = model.predict(X_test)
print("Độ chính xác khi sử dụng gt Cây Bayes thơ ngây và KFol",accuracy_score(thucte,dubao)*100)
#Độ chính xác 46.3768115942029