###Cây quyết định - KFold
X = dt.iloc[:,0:10]
y = dt.iloc[:,10:11]
kf= KFold(n_splits=70,shuffle=True,random_state=True)
for train_index, test_index in kf.split(X):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X.iloc[train_index,], X.iloc[test_index,]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=3, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)
y_pred = clf_gini.predict(X_test)
print("Độ chính xác cây quyết định khi sử dụng nghi thức phân chia dl KFold",accuracy_score(y_test,y_pred)*100)
print(confusion_matrix(y_test, y_pred))
###Cây Bayes thơ ngây - Hold-Out
X = dt.iloc[:,0:10]
y = dt.iloc[:,10:11]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)
model = GaussianNB()
model.fit(X_train, y_train.values.ravel())
thucte = y_test
dubao = model.predict(X_test)
print("Độ chính xác cây Bayes thơ ngây khi sử dụng nghi thức phân chia dl Hold-out",accuracy_score(thucte,dubao)*100)
cnf_matrix_gnb = confusion_matrix(thucte,dubao)
print(cnf_matrix_gnb)