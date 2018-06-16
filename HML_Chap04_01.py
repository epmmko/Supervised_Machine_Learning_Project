##LINEAR REGRESSION
import numpy as np                     #Uses np as alias of numpy
X=2*np.random.rand(100,1)              #Numpy array type(float64),(100x1)
y=4+3*X+np.random.rand(100,1)          #Predicted values between 0-1
import matplotlib.pyplot as plt
plt.plot(X,y,'ro')

X_b=np.c_[np.ones((100,1)),X]                              #Adds X_0=1 to each instance
theta_best=np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y) #Normal Equation. Array (2x1)

X_new=np.array([[0],[2]])             #Creates vector column [0 2]'
X_new_b=np.c_[np.ones((2,1)),X_new]   #First column full of 1
y_predict=X_new_b.dot(theta_best)     #Linear model prediction

plt.plot(X_new,y_predict,'r-',X,y,'b.')

#Using Scikit-Learn
from sklearn.linear_model import LinearRegression  #Imports function
lin_reg=LinearRegression()                         #Creates the function
lin_reg.fit(X,y)                                   #Execute the fitting
lin_reg.intercept_,lin_reg.coef_                   #Shows the values
lin_reg.predict(X_new)                             #Predicts

## GRADIENT DESCENT
#-Batch Gradient Descent
eta=0.1                                            #Learning rate
n_iteration=1000
m=100

theta=np.random.randn(2,1)                         #Random initialization

for iteration in range(n_iteration):
    gradients=2/m*X_b.T.dot(X_b.dot(theta)-y)      #Gradient equation
    theta=theta-eta*gradients
theta

#-Stochastic Gradient Descent
n_epochs=50
t0,t1=5,50                                         #Learning Schedule Parameter

def learning_schedule(t):                          #This function decreases the learning rate during each iteration
    return t0/(t+t1)

theta=np.random.randn(2,1)                         #Initial values for model parameters

for epoch in range(n_epochs):                      #For each epoch
    for i in range(m):                             #m times do:
        random_index=np.random.randint(m)          #Random index
        xi=X_b[random_index:random_index+1]        #Gets the row[index]
        yi=y[random_index:random_index+1]          #Gets the row[index]
        gradients=2*xi.T.dot(xi.dot(theta)-yi)     #Computes the gradient using ONLY 1 INSTANCE
        eta=learning_schedule(epoch*m+i)           #Learning rate value changes
        theta=theta-eta*gradients                  #New model parameters

theta

from sklearn.linear_model import SGDRegressor         #SGD Regressor
sgd_reg=SGDRegressor(n_iter=50,penalty=None,eta0=0.1) #Linear regression Stochastic using Scikit-Learn
sgd_reg.fit(X,y.ravel())
sgd_reg.intercept_,sgd_reg.coef_

#POLYNOMIAL REGRESSION
m=100
X=6*np.random.rand(m,1)-3
y=0.5*X**2+X+2+np.random.randn(m,1)
from sklearn.preprocessing import PolynomialFeatures
poly_features=PolynomialFeatures(degree=2,include_bias=False)
X_poly=poly_features.fit_transform(X)                          #Adds a new column with X^2 values

lin_reg=LinearRegression()
lin_reg.fit(X_poly,y)
lin_reg.intercept_,lin_reg.coef_

#LEARNING CURVES
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def plot_learning_curves(model,X,y):
    X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.2)
    train_errors,val_errors=[],[]
    for m in range(1,len(X_train)):
        model.fit(X_train[:m],y_train[:m])
        y_train_predict=model.predict(X_train[:m])
        y_val_predict=model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_predict,y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict,y_val))
    plt.plot(np.sqrt(train_errors),'r-+',linewidth=2,label='train')
    plt.plot(np.sqrt(val_errors),'b-',linewidth=3,label='val')
    
lin_reg=LinearRegression()
plot_learning_curves(lin_reg,X,y)

#Learning Curve - 10th Polynomial
from sklearn.pipeline import Pipeline
polynomial_regression=Pipeline([
    ('poly_features',PolynomialFeatures(degree=10,include_bias=False)),
    ('lin_reg',LinearRegression()),
])
plot_learning_curves(polynomial_regression,X,y)

#REGULARIZED LINEAR MODELS
#-Ridge
from sklearn.linear_model import Ridge       #Uses the l2 distance
ridge_reg=Ridge(alpha=1,solver='cholesky')   #Alpha:Regularization parameter
ridge_reg.fit(X,y)                           #Training the model
ridge_reg.predict([[1.5]])                   #Prediction
ridge_reg.intercept_
ridge_reg.coef_

#-Lasso
from sklearn.linear_model import Lasso      #Uses the l1 distance
lasso_reg=Lasso(alpha=0.1)                  #Alpha: Regularization parameter
lasso_reg.fit(X,y)                          #Training the model
lasso_reg.predict([[1.5]])                  #Prediction

#-Elastic Net
from sklearn.linear_model import ElasticNet          #Uses a combination of l1/L2 distance
elastic_net=ElasticNet(alpha=0.1,l1_ratio=0.5)       #Alpha: Regularization parameter
elastic_net.fit(X,y)                                 #Training the model
elastic_net.predict([[1.5]])                         #Prediction

#EARLY STOPPING
from sklearn.base import clone                                              #Function to copy the model
from sklearn.preprocessing import StandardScaler                            #Function to scale the data
poly_scaler=Pipeline([
        ('poly_features',PolynomialFeatures(degree=90,include_bias=False)), #First, performs a polynomial regression
        ('std_scaler',StandardScaler())])                                   #Then, scales the values
X_train_poly_scaled=poly_scaler.fit_transform(X_train)                     #Executes the transformation
X_val_poly_scaled=poly_scaler.transform(X_val)                              

sgd_reg=SGDRegressor(n_iter=1,warm_start=True,penalty=None,                 #ONLY 1 iteration is performed each time, Warm Start let the fit function use the previous results too.
                     learning_rate='constant',eta0=0.0005)

minimum_val_error=float('inf')
best_epoch=None
best_model=None
for epoch in range(1000):
    sgd_reg.fit(X_train_poly_scaled,y_train)
    y_val_predict=sgd_reg.predict(X_val_poly_scaled)
    val_error=mean_squared_error(y_val_predict,y_val)
    if val_error<minimum_val_error:
        minimum_val_error=val_error
        best_epoch=epoch
        best_model=clone(sgd_reg)


#DECISION BOUNDARIES -Logistic Regression
from sklearn import datasets             #Module with the datasets
import numpy as np     
import matplotlib.pyplot as plt

iris=datasets.load_iris()                #Types of Iris
#This data set contains Sepal Length, Sepal Width, Petal Length and Petal Width
list(iris.keys())
X=iris['data'][:,3:]                      #Select all rows and column[3] of Data Array
y=(iris['target']==2).astype(np.int)      #from data, 2 correspond to Virginica. This create 1 for True and 0 for False

from sklearn.linear_model import LogisticRegression
log_reg=LogisticRegression()
log_reg.fit(X,y)                          #Training

#Plot of probability for flower with petal width from 0 ->3cm
X_new=np.linspace(0,3,1000).reshape(-1,1) #Creates (Unknown rows,1Column), 1000 values from 0 to 3
y_proba=log_reg.predict_proba(X_new)      #Creates [0] column with False, and [1] column with True Probability
plt.plot(X_new,y_proba[:,1],'g-',label='Iris-Virginica')
plt.plot(X_new,y_proba[:,0],'b--',label='No Iris-Virginica')
log_reg.predict([[1.7],[1.5]])

#SOFTMAX REGRESSION
X=iris['data'][:,(2,3)]                 #Petal length, Petal Column
y=iris['target']                        #Target: 3 types of iris
softmax_reg=LogisticRegression(multi_class='multinomial',solver='lbfgs',C=10) #Multinomial indicates the use of many classes
softmax_reg.fit(X,y)
softmax_reg.predict([[5,2]])
softmax_reg.predict_proba([[5,2]])
