from sklearn.svm import SVC                    #Support Vector Machine Library/Classifier
from sklearn import datasets                   #Datasets library
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier #Stochastic Gradient Descent classifier deals with training instances independetly, one at a time.
from HML_Chap05_01 import plot_predictions,plot_dataset

#Linearly separable dataset
iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]                    # petal length (Column2), petal width (column3)
y = iris["target"]

setosa_or_versicolor = (y == 0) | (y == 1)     #Condition1 or Condition2
X = X[setosa_or_versicolor]                    #New data excludes y==2 virginica
y = y[setosa_or_versicolor]


svm_clfs = []
linear_svm_clf=Pipeline([
        ('scaler',StandardScaler()),
        ('linear_svc',LinearSVC(C=1,loss='hinge',random_state=42)),
        ])
linear_svm_clf.fit(X,y)
svm_clfs.append(linear_svm_clf)

linear_kernel_svm_clf=Pipeline([
        ('scaler',StandardScaler()),
        ('linear_kernel_clf',SVC(kernel='linear',C=1,random_state=42))
        ])
linear_kernel_svm_clf.fit(X,y)
svm_clfs.append(linear_kernel_svm_clf)

linear_sgd_clf=Pipeline([
        ('scaler',StandardScaler()),
        ('linear_sgd_clf',SGDClassifier(alpha=1,random_state=42)) #Create the classifier function, the 42 let us to obtain the same result everytime.
        ])
linear_sgd_clf.fit(X,y)
svm_clfs.append(linear_sgd_clf)
classifier=['LinearSVC','SVC(Kernel)','SGDClassifier']

plt.figure(figsize=(15,10))

for i, svm_clf in enumerate(svm_clfs):
    plt.subplot(311 + i)
    plot_predictions(svm_clf, [0, 6, 0, 2])
    plot_dataset(X, y, [0, 6, 0, 2])
    plt.title(classifier[i], fontsize=16)

plt.show()