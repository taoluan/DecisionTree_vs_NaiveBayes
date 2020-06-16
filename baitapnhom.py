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
x = dt.iloc[:,0:10]
y = dt.iloc[:,10:11]
#print(len(x))
#plt.scatter(x,y)
#plt.show()
#print(x)
#print(temp.count('g'))
#print(len(dt.fLength))
#print(len(np.unique(y)))
#Bieu do
def showbieudo(data1,data2,title):
    labels = ['Lan 1', 'Lan 2', 'Lan 3', 'Lan 4', 'Lan 5', 'Lan 6',  'Lan 7',  'Lan 8',  'Lan 9',  'Lan 10']
    x = np.arange(len(labels)) 
    width = 0.35 
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, data1, width, label='Cay Quyet Dinh')
    rects2 = ax.bar(x + width/2, data2, width, label='Cay Bayes')
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    autolabel(rects1)
    autolabel(rects2)
    fig.tight_layout()
    plt.show()
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
    i = 0
    arr_accqd = []
    arr_bayes = []
    model = GaussianNB()
    kf= KFold(n_splits=10,shuffle=True,random_state=True)
    for train_index, test_index in kf.split(X):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X.iloc[train_index,], X.iloc[test_index,]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        #cay quyet dinh
        nghithuc_gini = DecisionTreeClassifier(criterion = "gini", random_state = None,max_depth=8, min_samples_leaf=1)
        nghithuc_gini.fit(X_train, y_train)
        y_pred = nghithuc_gini.predict(X_test)
        accuracy_cqd += accuracy_score(y_test,y_pred)
        arr_accqd.append(round((accuracy_score(y_test,y_pred)*100),2))
        #cay bayes
        model.fit(X_train, y_train.values.ravel())
        thucte = y_test
        dubao = model.predict(X_test)
        accuracy_bayes += accuracy_score(thucte,dubao)
        arr_bayes.append(round((accuracy_score(thucte,dubao)*100),2))
        #print("Lan",i+1,"Accuracy cua cay quyet dinh =",accuracy_score(y_test,y_pred)*100,"Va cua cay bayes =",accuracy_score(thucte,dubao)*100,)
        i += 1
    #print("TOTAL: Accuracy Va Cayquyetdinh-kfold =",(accuracy_cqd/10)*100," Va Bayes-kfold =",(accuracy_bayes/10)*100)
    #showbieudo(arr_accqd,arr_bayes,"Nghi thuc Kfold")
    return [round((accuracy_cqd/10)*100,2),round((accuracy_bayes/10)*100,2)]
def cayquyetdinh_vs_caybayes_holdout(x,y):
    model = GaussianNB()
    accuracy_bayes = 0
    accuracy_cqd = 0
    arr_accqd = []
    arr_bayes = []
    for i in range(0,10):
        X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=1/3.0,shuffle=True,random_state=None)
        nghithuc_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=9, min_samples_leaf=1)
        nghithuc_gini.fit(X_train, y_train)
        y_pred = nghithuc_gini.predict(X_test)
        accuracy_cqd += accuracy_score(y_test,y_pred)
        arr_accqd.append(round((accuracy_score(y_test,y_pred)*100),2))
        model.fit(X_train, y_train.values.ravel())
        thucte = y_test
        dubao = model.predict(X_test)
        accuracy_bayes += accuracy_score(thucte,dubao)
        arr_bayes.append(round((accuracy_score(thucte,dubao)*100),2))
       #print("Lan",i+1,"Accuracy cua cay quyet dinh =",accuracy_score(y_test,y_pred)*100,"Va cua cay bayes =",accuracy_score(thucte,dubao)*100)
    #print("TOTAL: Accuracy Va Cayquyetdinh-HouldOut =",(accuracy_cqd/10)*100," Va Bayes-HouldOut     =",(accuracy_bayes/10)*100)
    #showbieudo(arr_accqd,arr_bayes,"Nghi thuc Hold-Out")
    return [round((accuracy_cqd/10)*100,2),round((accuracy_bayes/10)*100,2)]
def danhgia_holfout_kfold(x,y):
    accuracy_kf = cayquyetdinh_vs_caybayes_kfold(x,y)
    accuracy_ho = cayquyetdinh_vs_caybayes_holdout(x,y)
    print(accuracy_ho,accuracy_kf)
    data1 = [accuracy_ho[0],accuracy_kf[0]]
    data2 = [accuracy_ho[1],accuracy_kf[1]]
    labels = ['Hould Out','Kfold']
    x = np.arange(len(labels)) 
    width = 0.4
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, data1, width, label='Cay Quyet Dinh')
    rects2 = ax.bar(x + width/2, data2, width, label='Cay Bayes')
    ax.set_ylabel('Accuracy')
    ax.set_title('Do chinh xac cua Cay Quyet Dinh va Cay Bayes')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    autolabel(rects1)
    autolabel(rects2)
    fig.tight_layout()
    plt.show()
    print(accuracy_ho)
#cayquyetdinh_holdout_gini(x,y)
#cayquyetdinh_holdout_entropy(x,y)
#caybayes_kfold(x,y)
#2-3 thuoc tinh (fLength,fSize,fConc1) 7-10 du lieu xay dung mo hinh
#16.9503,2.3385,0.4151,g
#17.5606,2.3385,0.3096,g
#17.2692,2.2504,0.3118,g
#27.3481,2.433,0.2417,h
#22.3554,2.6349,0.3118,h
#21.873,2.433,0.3118,h
#26.3485,2.3261,0.4151,h
#28.0883,2.2504,0.3174,g
#cayquyetdinh_vs_caybayes_holdout(x,y)
#cayquyetdinh_vs_caybayes_kfold(x,y)
danhgia_holfout_kfold(x,y)