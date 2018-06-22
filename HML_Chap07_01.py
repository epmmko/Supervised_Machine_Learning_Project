import time
start = time.time()

# VOTING CLASSIFIER
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

X,y=make_moons(n_samples=500,noise=0.3,random_state=42)                         #DataSets
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42)

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

log_clf=LogisticRegression(random_state=42)                                    #Three Model Classifiers 
rnd_clf=RandomForestClassifier(random_state=42)
svm_clf=SVC(random_state=42,probability=True)

voting_clf=VotingClassifier(                                                   #Majority rule classifier
        estimators=[('lr',log_clf),('rf',rnd_clf),('svc',svm_clf)],            #List of (string,estimator) tuples
        voting='soft')                                                         #hard or soft

from sklearn.metrics import accuracy_score
for clf in (log_clf,rnd_clf,svm_clf,voting_clf):
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    print(clf.__class__.__name__,accuracy_score(y_test,y_pred))                #Accuracy is improved when the classifiers are grouped with voting

#BAGGING AND PASTING
from sklearn.ensemble import BaggingClassifier                                 #Subtraining sets with replacement (bootstrap)
from sklearn.tree import DecisionTreeClassifier                                #Base classifier
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

bag_clf=BaggingClassifier(
        DecisionTreeClassifier(),n_estimators=500,                             #500 minisets
        max_samples=100,bootstrap=True,n_jobs=1,                               #100 instances, with replacement and parallel jobs
        oob_score=True)                                                        #Activate Out-of-bag evaluation. (Instances not selected in the minisets)
bag_clf.fit(X_train,y_train)
print(bag_clf.oob_score_)
y_pred=bag_clf.predict(X_test)
bag_clf.oob_decision_function_                                                 #Returns the probability of belonging to a class for each instance

def plot_decision_boundary(clf, X, y, axes=[-1.5, 2.5, -1, 1.5], alpha=0.5, contour=True):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    if contour:
        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", alpha=alpha)
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", alpha=alpha)
    plt.axis(axes)
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.ylabel(r"$x_2$", fontsize=18, rotation=0)

plt.figure(figsize=(11,4))
plot_decision_boundary(bag_clf, X, y)
plt.title("Decision Trees with Bagging", fontsize=14)
plt.show()


#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier                            #An optimized bagging algorithm
rnd_clf=RandomForestClassifier(n_estimators=500,max_leaf_nodes=16,n_jobs=1)    #500 tree classifier, max_leaf_nodes regularizes
rnd_clf.fit(X_train,y_train)
y_pred_rf=rnd_clf.predict(X_test)

from sklearn.datasets import load_iris
iris=load_iris()
rnd_clf=RandomForestClassifier(n_estimators=500,n_jobs=1)
rnd_clf.fit(iris['data'],iris['target'])
for name,score in zip(iris['feature_names'],rnd_clf.feature_importances_):     #Feature importances. In this case, Petal length and width have the highest importance
    print(name,score)

#BOOSTING
from sklearn.ensemble import AdaBoostClassifier
#It is a meta-estimator that begins by fitting a classifier on the original dataset and then fits additional copies of the classifier 
#on the same dataset but where the weights of the incorrectly classified instances are adjusted such thay subsequent classifier focus
#more on difficult cases
ada_clf=AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=1),n_estimators=200,                  #DecisionTreeClassifier is the base estimator, 200 estimators are used
        algorithm='SAMME.R',learning_rate=0.5,random_state=42)                 #SAMME.R relies on probabilities calculations
ada_clf.fit(X_train,y_train)
plot_decision_boundary(ada_clf,X,y)

start = time.time()
#Gradient Boosting
np.random.seed(42)
X_new=np.random.rand(100,1)-0.5                                                    #Quadratic random dataset
y_new=3*X_new[:,0]**2+0.05*np.random.randn(100)
from sklearn.tree import DecisionTreeRegressor
tree_reg1=DecisionTreeRegressor(max_depth=2,random_state=42)                   #First estimator
tree_reg1.fit(X_new,y_new)
y2=y_new-tree_reg1.predict(X_new)                                                      #Residuals calculation
tree_reg2=DecisionTreeRegressor(max_depth=2,random_state=42)                   #Second estimator trained with the residuals
tree_reg2.fit(X_new,y2)
y3=y2-tree_reg2.predict(X_new)
tree_reg3=DecisionTreeRegressor(max_depth=2,random_state=42)                   #Third estimator trained with the residuals of the residuals
tree_reg3.fit(X_new,y3)

y_pred=sum(tree.predict(X_new) for tree in (tree_reg1,tree_reg2,tree_reg3))    #The final prediction is the sum of all predictions

from sklearn.ensemble import GradientBoostingRegressor                         #Gradient boosting class
gbrt=GradientBoostingRegressor(max_depth=2,n_estimators=3,learning_rate=1)     #3 Tree with max_depth of 2.
#Learning rate: If we set it to a low value, such as 0.1, we will need more trees in the ensemble to fit the trainin set, but the
#predictions will usually generalize better
gbrt.fit(X_new,y_new)

gbrt_slow=GradientBoostingRegressor(max_depth=2,n_estimators=200,learning_rate=0.1)
gbrt_slow.fit(X_new,y_new)

def plot_predictions(regressors, X, y, axes, label=None, style="r-", data_style="b.", data_label=None):
    x1 = np.linspace(axes[0], axes[1], 500)
    y_pred = sum(regressor.predict(x1.reshape(-1, 1)) for regressor in regressors)
    plt.plot(X[:, 0], y, data_style, label=data_label)
    plt.plot(x1, y_pred, style, linewidth=2, label=label)
    if label or data_label:
        plt.legend(loc="upper center", fontsize=16)
    plt.axis(axes)

plt.figure(figsize=(11,4))

plt.subplot(121)
plot_predictions([gbrt], X_new, y_new, axes=[-0.5, 0.5, -0.1, 0.8], label="Ensemble predictions")
plt.title("learning_rate={}, n_estimators={}".format(gbrt.learning_rate, gbrt.n_estimators), fontsize=14)

plt.subplot(122)
plot_predictions([gbrt_slow], X_new, y_new, axes=[-0.5, 0.5, -0.1, 0.8])
plt.title("learning_rate={}, n_estimators={}".format(gbrt_slow.learning_rate, gbrt_slow.n_estimators), fontsize=14)
plt.show()

#Optimal number of trees
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_train,X_val,y_train,y_val=train_test_split(X_new,y_new,random_state=49)

gbrt=GradientBoostingRegressor(max_depth=2,n_estimators=120,random_state=42)   #120 estimators with 2 level of depth
gbrt.fit(X_train, y_train)

errors=[mean_squared_error(y_val, y_pred)                                      #Measure the validation error
          for y_pred in gbrt.staged_predict(X_val)]                            #Staged_predict returns an iterator over the prediction made by the ensemble at each stage of training
bst_n_estimators=np.argmin(errors)                                             #Best number of tree is the one who produces the least error
min_error=np.min(errors)
gbrt_best=GradientBoostingRegressor(max_depth=2,n_estimators=bst_n_estimators, random_state=42)
gbrt_best.fit(X_train, y_train)

plt.figure(figsize=(11, 4))

plt.subplot(121)
plt.plot(errors, "b.-")                                                        #Plot Errors (blue line)
plt.plot([bst_n_estimators, bst_n_estimators], [0, min_error], "k--")          #Vertical line (dotted and black) in optimal number of trees
plt.plot([0, 120], [min_error, min_error], "k--")                              #Horizontal line (dotted and black) in minimal error
plt.plot(bst_n_estimators, min_error, "ko")
plt.text(bst_n_estimators, min_error*1.2, "Minimum", ha="center", fontsize=14) #Legend
plt.axis([0, 120, 0, 0.01])
plt.xlabel("Number of trees")
plt.title("Validation error", fontsize=14)

plt.subplot(122)
plot_predictions([gbrt_best], X_new, y_new, axes=[-0.5, 0.5, -0.1, 0.8])
plt.title("Best model (%d trees)" % bst_n_estimators, fontsize=14)
plt.show()

end = time.time()
print(end - start)