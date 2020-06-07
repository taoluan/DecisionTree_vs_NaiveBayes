import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 
import sklearn
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
dt = pd.read_csv("Housing_2019.csv",index_col=0)
dt.iloc[2:4,]
x = dt.iloc[:,[1,2,3,4,10]]
#x1 = x.iloc[1:5,]
y = dt.price
print (dt.lotsize)
print (y)
plt.scatter(dt.lotsize,y)
plt.show()
