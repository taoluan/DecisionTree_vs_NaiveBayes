import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
thuoctinh = np.array([[0,0,1,1],[0,1,0,1]])
nhan =  np.array([0,0,0,1])
thuoctinh = thuoctinh.T
#print(thuoctinh[:,1])
colormap = np.array(['red','green'])
#plt.axis([0,1.2,0,1.2])
#plt.scatter(thuoctinh[:,0],thuoctinh[:,1],c = colormap[nhan],s=150)
#plt.xlabel("Gia tri x1")
#plt.ylabel("Gia tri x2")
#plt.show()
def my_perceptron(x,y,eta,lanlap):
    n=len(x[0,])
    m = len(x[:,0])
    print("m= ",m," va n= ",n)
    #w0 = random.random() #khoi tao ngau nhien
    #w = np.random.random(n) #khoi tao ngau nhien gia tri w
    w0 = -0.2
    w =(0.5,0.5)
    print(" w0 = ",w0)
    print(" w = ",w)
    for i in range(0,lanlap):
        print("Lan lap ... ",i)
        for j in range(0,m):
            gx = w0 + sum(x[j,]*w)
            print("gx = ",gx)
            if(gx > 0):
                op = 1 
            else:
                op = 0
            w0 = w0 + eta*(y[j]-op)
            w = w + eta*(y[j]-op)*x[j,]
            print("w0 = ", w0)
            print("w = ",w)
    return (w0,w)
#my_perceptron(thuoctinh,nhan,0.15,2)
data = pd.read_csv("data_per.csv",delimiter=",")
value_tt = data.iloc[:,0:5]
value_nhan = data.iloc[:,5:6]
#print(value_nhan)
x_train,x_test,y_train,y_test = train_test_split(value_tt,value_nhan,test_size=1/3.0,random_state=100)
net = Perceptron()
net.fit(x_train,y_train.values.ravel())
value_dubao = net.predict(x_test)
#print(net.coef_) # gia tri Op
#print(net.score(x_train,y_train.values.ravel())*100)
print(accuracy_score(y_test,value_dubao)*100)
