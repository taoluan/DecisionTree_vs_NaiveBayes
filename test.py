import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
import numpy as np
import pylab as pl

dt=pd.read_csv("winequality-red.csv",delimiter=";")
x = dt.iloc[:,0:11]
y = dt.iloc[:,11:12]
#print(len(dt)) 
# Cau a: co 1599 phan tu , nhan {0,1,2,3,4,5,6,7,8,9}
# Cau b:
X_train,X_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=100)
#print(len(X_test))
#print(len(X_train))
#Cau c: Xay dung mo hinh x_train = 1279 , x_test = 320
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=3, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)

clf_gini.score(X_train, y_train)


#du doan va danh gia
y_pred = clf_gini.predict(X_test)
y_test
 
cm = confusion_matrix(y_test, y_pred)

print("Accuracy is",accuracy_score(y_test,y_pred)*100)
print(confusion_matrix(y_test, y_pred))
pl.matshow(cm)
pl.title('Confusion matrix of the classifier')
pl.colorbar()
pl.show()