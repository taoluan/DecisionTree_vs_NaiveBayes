import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


dulieu = pd.read_csv("housing_RT.csv" , index_col=0)
dulieu.ix[1:5,]
dulieu.ilox[1:5,]


X_train,X_test,y_train,y_test = train_test_split( dulieu.ix[:,1:5],dulieu.ix[:,0], test_size=1/3.0, random_state=100) 
X_train[1:5,]
X_test[1:5] 

X_train,X_test,y_train,y_test = train_test_split( dulieu.ix[:,1:5],dulieu.ix[:,0], test_size=1/3.0, random_state=100)
X_train[1:5,] 
X_test[1:5] 
y_pred = regressor.predict(X_test) 
y_test[1:5] 
y_pred[1:5]
print("Accuracy is ",accuracy_score(y_test,y_pred)*100)
  
mean_squared_error(y_test,y_pred)
err 
np.sqrt(err)