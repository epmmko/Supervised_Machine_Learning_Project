##HML-SELFHOMEWORK_EXERCISE_12
#Implement Batch Gradient Descent with early stopping for Softmax Regression
# (without using Scikit-Learn).

#DECISION BOUNDARIES -Logistic Regression
from sklearn import datasets            #Module with the datasets
import numpy as np     
import matplotlib.pyplot as plt

iris=datasets.load_iris()               #Types of Iris

X=iris['data'][:,(2,3)]                 #Petal length, Petal Column
y=iris['target']                        #Target: 3 types of iris

shuffle_index=np.random.permutation(150)#Perform a random permutation of the number 0-60,000
X,y=X[shuffle_index],y[shuffle_index]   #Shuffles the train_set to guarantee that all cross-validation fikds will be similar

#Splitting in Train and Test Set
X_train,X_val,y_train,y_val=X[:100],X[100:],y[:100],y[100:]
#The first 100 instances are for trainning, the remaining is for testing


#EARLY STOPPING
from sklearn.base import clone                                              #Function to copy the model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler                            #Function to scale the data
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error


poly_scaler=Pipeline([
    ('poly_features',PolynomialFeatures(degree=2,include_bias=False)), #First, performs a polynomial regression
    ('std_scaler',StandardScaler())])                                   #Then, scales the values
X_train_poly_scaled=poly_scaler.fit_transform(X_train)                     #Executes the transformation
X_val_poly_scaled=poly_scaler.transform(X_val)                              

#SOFTMAX REGRESSION
from sklearn.linear_model import LogisticRegression
softmax_reg=LogisticRegression(multi_class='multinomial',solver='lbfgs',C=10,
                               max_iter=1,warm_start=True)                 #ONLY 1 iteration is performed each time, Warm Start let the fit function use the previous results too.
                            
minimum_val_error=float('inf')
best_epoch=None
best_model=None
for epoch in range(500):
    softmax_reg.fit(X_train_poly_scaled,y_train)
    y_val_predict=softmax_reg.predict(X_val_poly_scaled)
    val_error=mean_squared_error(y_val_predict,y_val)
    if val_error<minimum_val_error:
        minimum_val_error=val_error
        best_epoch=epoch
        best_model=clone(softmax_reg)

best_model.fit(X_train,y_train)
best_model.predict([[5,2]])
best_model.predict_proba([[5,2]])
