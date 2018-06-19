from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')

X,y=mnist['data'],mnist['target']
X.shape                           #Rows are intances, Columns are features (784 features=28x28pixeles). Each pixel (0:white -> 255:black)
y.shape                           #70,000 numbers between 0-9

import numpy as np                           #Uses np as alias of numpy
shuffle_index=np.random.permutation(70000)   #Perform a random permutation of the number 0-60,000
X,y=X[shuffle_index],y[shuffle_index]        #Shuffles the train_set to guarantee that all cross-validation fikds will be similar

#Splitting in Train and Test Set
X_train,X_test,y_train,y_test=X[:60000],X[60000:],y[:60000],y[60000:]
#The first 60,000 instances are for trainning, the remaining is for testing

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train.astype(np.float32))
X_test_scaled=scaler.fit_transform(X_test.astype(np.float32))

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal, uniform
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

linear_svc_clf=SVC()
param_distribution={"gamma":reciprocal(0.001,0.1),"C":uniform(1,10)}
rnd_search_cv=RandomizedSearchCV(linear_svc_clf,param_distribution,n_iter=10,verbose=2,n_jobs=1)

rnd_search_cv.fit(X_train_scaled[:10000],y_train[:10000])
rnd_search_cv.best_estimator_
print('done')
rnd_search_cv.best_estimator_.fit(X_train,y_train)
y_pred=rnd_search_cv.best_estimator_.predict(X_train)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)