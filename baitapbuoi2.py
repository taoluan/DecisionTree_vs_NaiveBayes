import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
import numpy as np
# Cau a: 
dt=pd.read_csv("winequality-red.csv",delimiter=";")
x = dt.iloc[:,0:11]
y = dt.iloc[:,11:12]
#print(x)
#print(len(dt)) 
#print(np.unique(y))
#np.unique(dt.quality)
#dt.quality.value_counts()
# Cau b:co 1599 phan tu , nhan {3,4,5,6,7,8}
#Cau c: Xay dung mo hinh x_train = 1279 , x_test = 320
#X_train,X_test,y_train,y_test = train_test_split(x,y, test_size=1.0/10.0, random_state=100)
#print(len(y_test))
#print(np.unique(y_test))
#print(len(X_train))
#cau d: xay dung mo hinh
#clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth=7, min_samples_split=5)
#clf_entropy.fit(X_train, y_train)


#cau f: du doan va danh gia
#y_pred = clf_entropy.predict(X_test)
#y_test
#do chinh xac 100
#print("Accuracy is all",accuracy_score(y_test,y_pred)*100)
#print(confusion_matrix(y_test, y_pred))
#cau e do chinh xac 100 
#x = dt.iloc[0:9,1:11]
#y = dt.iloc[0:9,11:12]
#X_train,X_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=100)
#clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
#                               max_depth=3, min_samples_leaf=5)
#clf_gini.fit(X_train, y_train)
#y_pred = clf_gini.predict(X_test)
#y_test
#print("Accuracy is the first 9 elements",accuracy_score(y_test,y_pred)*100)
#print(confusion_matrix(y_test, y_pred))
####Câu  2
X =[[180,15,0],
          [167,42,1],
          [136,35,1],
          [174,15,0],
          [141,28,1]]
Y =['Nam','Nu','Nu','Nam','Nu']
#print(np.unique(Y))
X_train,X_test,y_train,y_test = train_test_split(X,Y, test_size=1/3.0)
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=3, min_samples_leaf=1)
clf_gini.fit(X_train, y_train)
y_pred = clf_gini.predict(X_test)
dudoan = clf_gini.predict([[133,37,1]])
print(dudoan)

print("Do chinh xac cua du lieu nhan nam(1) vs nu(0) = ",accuracy_score(y_test,y_pred)*100)
print(confusion_matrix(y_test, y_pred))