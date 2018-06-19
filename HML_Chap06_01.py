from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from  sklearn.tree  import export_graphviz
import pydot
import numpy as np
import os
import matplotlib.pyplot as plt

iris=load_iris()
X=iris.data[:,2:]                            #Since column2 until the end. Petal length and width
y=iris.target

tree_clf=DecisionTreeClassifier(max_depth=2) #Maximum depth of the tree
tree_clf.fit(X,y)                            #Training the model

export_graphviz(                             #exports the image
        tree_clf,                            #Classifier
        out_file='iris_tree.dot',            #File name
        feature_names=iris.feature_names[2:],#Names of each feature
        class_names=iris.target_names,       #Names of each of the target classes in ascending numerical order 
        rounded=True,                        #Draws node boxes with rounded corners and use Helvetica fonts instead of Times-Roman
        filled=True                          #Paints nodes to indicate majority class for classification
    )

os.environ["PATH"] += os.pathsep + r'C:\Users\victo\Anaconda3\envs\datascience\Library\bin\graphviz' #Solves problem with the dot.exe path
(graph,)=pydot.graph_from_dot_file('iris_tree.dot')
graph.write_png('iris_tree.png')             #Converts dot file to png

from matplotlib.colors import ListedColormap

def plot_decision_boundary(clf, X, y, axes=[0, 7.5, 0, 3], iris=True, legend=False, plot_training=True):
    x1s = np.linspace(axes[0], axes[1], 100)      #Petal length axe
    x2s = np.linspace(axes[2], axes[3], 100)      #Petal width axe
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]         #Pairs with coordenates
    y_pred = clf.predict(X_new).reshape(x1.shape) #Prediction 100x100
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap) #Contour with decision boundaries
    if not iris:
        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    if plot_training:
        plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", label="Iris-Setosa")
        plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", label="Iris-Versicolor")
        plt.plot(X[:, 0][y==2], X[:, 1][y==2], "g^", label="Iris-Virginica")
        plt.axis(axes)
    if iris:
        plt.xlabel("Petal length", fontsize=14)
        plt.ylabel("Petal width", fontsize=14)
    else:
        plt.xlabel(r"$x_1$", fontsize=18)
        plt.ylabel(r"$x_2$", fontsize=18, rotation=0)
    if legend:
        plt.legend(loc="lower right", fontsize=14)

plt.figure(figsize=(8, 4))
plot_decision_boundary(tree_clf, X, y)
plt.plot([2.45, 2.45], [0, 3], "k-", linewidth=2)       #Plot Depth=0
plt.plot([2.45, 7.5], [1.75, 1.75], "k--", linewidth=2) #Plot Depth=1
plt.plot([4.95, 4.95], [0, 1.75], "k:", linewidth=2)    #Plot Depth=2
plt.plot([4.85, 4.85], [1.75, 3], "k:", linewidth=2)
plt.text(1.40, 1.0, "Depth=0", fontsize=15)
plt.text(3.2, 1.80, "Depth=1", fontsize=13)
plt.text(4.05, 0.5, "(Depth=2)", fontsize=11)
plt.show()

tree_clf.predict_proba([[5,1.5]])                      #Probability prediction for the 3 classes
tree_clf.predict([[5,1.5]])                            #Class prediction

#REGULARIZATION HYPERPARAMETER
from sklearn.datasets import make_moons
Xm, ym = make_moons(n_samples=100, noise=0.25, random_state=53)             #Make moon instances

deep_tree_clf1 = DecisionTreeClassifier(random_state=42)                    #Model 1, No restricted
deep_tree_clf2 = DecisionTreeClassifier(min_samples_leaf=4, random_state=42)#Model 2, regularized
deep_tree_clf1.fit(Xm, ym)                                                  #Training
deep_tree_clf2.fit(Xm, ym)

plt.figure(figsize=(11, 4))
plt.subplot(121)
plot_decision_boundary(deep_tree_clf1, Xm, ym, axes=[-1.5, 2.5, -1, 1.5], iris=False)
plt.title("No restrictions", fontsize=16)
plt.subplot(122)
plot_decision_boundary(deep_tree_clf2, Xm, ym, axes=[-1.5, 2.5, -1, 1.5], iris=False)
plt.title("min_samples_leaf = {}".format(deep_tree_clf2.min_samples_leaf), fontsize=14)
plt.show()

#REGRESSION TREE
# Quadratic training set + noise
np.random.seed(42)                 #This let us to obtain the same results from the random function
m = 200                            #Number of instances
X = np.random.rand(m, 1)           #200 Random values between 0-1
y = 4 * (X - 0.5) ** 2             #Quadratic function
y = y + np.random.randn(m, 1) / 10 #Noise

from sklearn.tree import DecisionTreeRegressor
tree_reg1 = DecisionTreeRegressor(random_state=42, max_depth=2) #Regressor with max depth 2
tree_reg2 = DecisionTreeRegressor(random_state=42, max_depth=3) #Regressor with max depth 3
tree_reg1.fit(X, y)                                             #Training
tree_reg2.fit(X, y)

def plot_regression_predictions(tree_reg, X, y, axes=[0, 1, -0.2, 1], ylabel="$y$"):
    x1 = np.linspace(axes[0], axes[1], 500).reshape(-1, 1)
    y_pred = tree_reg.predict(x1)
    plt.axis(axes)
    plt.xlabel("$x_1$", fontsize=18)
    if ylabel:
        plt.ylabel(ylabel, fontsize=18, rotation=0)
    plt.plot(X, y, "b.")
    plt.plot(x1, y_pred, "r.-", linewidth=2, label=r"$\hat{y}$")

plt.figure(figsize=(11, 4))
plt.subplot(121)
plot_regression_predictions(tree_reg1, X, y)
for split, style in ((0.1973, "k-"), (0.0917, "k--"), (0.7718, "k--")):
    plt.plot([split, split], [-0.2, 1], style, linewidth=2)
plt.text(0.21, 0.65, "Depth=0", fontsize=15)
plt.text(0.01, 0.2, "Depth=1", fontsize=13)
plt.text(0.65, 0.8, "Depth=1", fontsize=13)
plt.legend(loc="upper center", fontsize=18)
plt.title("max_depth=2", fontsize=14)

plt.subplot(122)
plot_regression_predictions(tree_reg2, X, y, ylabel=None)
for split, style in ((0.1973, "k-"), (0.0917, "k--"), (0.7718, "k--")):
    plt.plot([split, split], [-0.2, 1], style, linewidth=2)
for split in (0.0458, 0.1298, 0.2873, 0.9040):
    plt.plot([split, split], [-0.2, 1], "k:", linewidth=1)
plt.text(0.3, 0.5, "Depth=2", fontsize=13)
plt.title("max_depth=3", fontsize=14)
plt.show()