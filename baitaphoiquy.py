import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 
import sklearn
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

dt = pd.read_csv("Housing_2019.csv",index_col=0)
x = dt.iloc[:,[1,2,3,4,10]]
y = dt.price

X_train,X_test,y_train,y_test = train_test_split(x,y, test_size=0.3, random_state=100)
len(X_train)#382

lm = linear_model.LinearRegression()
lm.fit(X_train,y_train)

#print(lm.intercept_) #-4703.035834269729
#print(lm.coef_) #[4.21647083e+00 3.69974604e+03 1.79388048e+04 7.12786323e+036.37681260e+03]


Y_pred = lm.predict(X_test)
#y_test


err = mean_squared_error(y_test, Y_pred)
#print(err)#341179713.3272553
np.sqrt(err)

#cau2
def LR2(X, Y, eta, lanlap, theta0, theta1):
    m = len(X)#so luong phan tu
    for i in range(0,lanlap):
        print("Lan lap: ", i)
        delta0 = 0
        delta1 = 0
        for j in range(0,m):
            h = theta0 + theta1*X[j]
            delta0 = delta0 + (Y[j]-h)*1
            delta1 = delta1 + (Y[j]-h)*X[j]
        theta0 = theta0 + (delta0/m)*eta
        theta1 = theta1 + (delta1/m)*eta
        #theta0 = theta0 + (delta0)*eta
        #theta1 = theta1 + (delta1)*eta
        print ("Lan lap",i, "gia tri theta0 = ",theta0)
        print ("Lan lap",i,  "gia tri theta1 = ",theta1)
    return [theta0,theta1]


X = np.array([1,2,4]).T
Y = np.array([2,3,6]).T

theta = LR2(X,Y,0.2,2,0,1)
YY = theta[0] +theta[1]*5
#du bao
XX = [0,3,5]
for i in range(0,3):
    YY = theta[0] + theta[1]*XX[i]
    print (YY)

#lan 0: 0.13777777777777775
#lan 1: 4.084444444444444
#lan 2: 6.7155555555555555



