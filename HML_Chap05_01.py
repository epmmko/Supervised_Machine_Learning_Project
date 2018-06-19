from sklearn.svm import SVC                    #Support Vector Machine Library/Classifier
from sklearn import datasets                   #Datasets library
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]                    # petal length (Column2), petal width (column3)
y = iris["target"]

setosa_or_versicolor = (y == 0) | (y == 1)     #Condition1 or Condition2
X = X[setosa_or_versicolor]                    #New data excludes y==2 virginica
y = y[setosa_or_versicolor]

# SVM Classifier model- Hard Margin
svm_clf = SVC(kernel="linear", C=float("inf"))
svm_clf.fit(X, y)                              #Training model
w=svm_clf.coef_[0]                             #Coef. of the model
b=svm_clf.intercept_[0]                        #intercept of the model
x0 = np.linspace(0,5.5, 200)                   #Vector 200x1 with feature0
decision_boundary = -w[0]/w[1] * x0 - b/w[1]   #Decision line calculation
svs = svm_clf.support_vectors_                 
plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors='#FFAAAA')
plt.plot(x0, decision_boundary, "k-", linewidth=2)
plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs")   #plot blue squares both features for y==1
plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo")   #plot yellow circles both features for y==0
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.axis([0, 5.5, 0, 2])

X=iris['data'][:,(2,3)]                       # petal length (Column2), petal width (column3)
y=(iris['target']==2).astype(np.float64)      #Iris-Virginica (0 for False, 1 for True)

svm_clf=Pipeline([
        ('scaler',StandardScaler()),
        ('linear_svc',LinearSVC(C=1,loss='hinge',random_state=42)),
        ])
svm_clf.fit(X,y)
svm_clf.predict([[5.5,1.7]])

#NONLINEAR SVM CLASSIFICATION
from sklearn.datasets import make_moons
X,y=make_moons(n_samples=100,noise=0.15,random_state=42)  #Sample data

def plot_dataset(X,y,axes):
    plt.plot(X[:,0][y==0],X[:,1][y==0],'bs')              #2D plot when y==0
    plt.plot(X[:,0][y==1],X[:,1][y==1],'g^')              #2D plot when y==1
    plt.axis(axes)
    plt.grid(True,which='both')                           #Grid on in both axes
    plt.xlabel(r'$x_1$',fontsize=20)
    plt.ylabel(r'$x_2$',fontsize=20,rotation=0)

plot_dataset(X,y,[-1.5,2.5,-1,1.5])
plt.show()

from sklearn.preprocessing import PolynomialFeatures      #Creation of new features
polynomial_svm_clf=Pipeline([                             #Creation of the pipeline
        ('poly_features',PolynomialFeatures(degree=3)),
        ('scaler',StandardScaler()),
        ('svm_clf',LinearSVC(C=10,loss='hinge'))
        ])
polynomial_svm_clf.fit(X,y)
polynomial_svm_clf.predict([[1,-0.5]])
polynomial_svm_clf.predict([[1,0.5]])

def plot_predictions(clf, axes):                           #Function for plotting
    x0s = np.linspace(axes[0], axes[1], 100)               #X0 axe from axes[0] to axes[1]
    x1s = np.linspace(axes[2], axes[3], 100)               #X1 axe from axes[2] to axes[3]
    x0, x1 = np.meshgrid(x0s, x1s)                         #Creates the grid
    X = np.c_[x0.ravel(), x1.ravel()]                      #Creates the vector of independent variables
    y_pred = clf.predict(X).reshape(x0.shape)              #Predicts
    y_decision = clf.decision_function(X).reshape(x0.shape)#Calculate the decision function line
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)
    plt.contourf(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)

plot_predictions(polynomial_svm_clf, [-1.5, 2.5, -1, 1.5])
plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
plt.show()

#POLYNOMIAL KERNEL
from sklearn.svm import SVC
#SVC apply the kernel trick. It makes possible to get the same result as if you
#added many polynomial features without having to add them.
#degree indicates the polynomial degree
#coef0 controls how much the model is influenced by high-degree polynomials
poly_kernel_svm_clf=Pipeline([
        ('scaleer',StandardScaler()),
        ('svm_clf',SVC(kernel='poly',degree=3,coef0=1,C=5))
        ])
poly_kernel_svm_clf.fit(X,y)
poly100_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="poly", degree=10, coef0=100, C=5))
    ])
poly100_kernel_svm_clf.fit(X, y)

plt.figure(figsize=(11, 4))
#Subplot 1 row, 2 columns, #position
plt.subplot(121)
plot_predictions(poly_kernel_svm_clf, [-1.5, 2.5, -1, 1.5])
plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
plt.title(r"$d=3, r=1, C=5$", fontsize=18)

plt.subplot(122)
plot_predictions(poly100_kernel_svm_clf, [-1.5, 2.5, -1, 1.5])
plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
plt.title(r"$d=10, r=100, C=5$", fontsize=18)

#GAUSSIAN RBF KERNEL
#Radial Basis Function: Creates a landmark in each intance and then estimates
#the similarity of each other intances. Creates a set of m instances and m features
rbf_kernel_svm_clf=Pipeline([
        ('scaler',StandardScaler()),
        ('svm_clf',SVC(kernel='rbf',gamma=5,C=0.001))
        ])
#Gamma: Increasing gamma makes the bell-shape narrower. Instances' range of influence is smaller. The decision bounder is more irregular
#Similar to Gamma.
rbf_kernel_svm_clf.fit(X,y)
plot_predictions(rbf_kernel_svm_clf, [-1.5, 2.5, -1, 1.5])
plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
plt.show()