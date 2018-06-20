import time
start = time.time()
#PART EXERCISE 07
from sklearn.datasets import make_moons
X,y=make_moons(n_samples=10000, noise=0.4, random_state=42)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
from sklearn.tree import DecisionTreeClassifier
param_grid={'max_depth':[3,5,7,10,15],'min_samples_leaf':[1,2,3]} #Run DecisionTreeClassifier for each combination of max_depth and min_sample_leaf
from sklearn.model_selection import GridSearchCV                     #Finds the best combination of hyperparameters using cross-vaidation
grid_search=GridSearchCV(DecisionTreeClassifier(random_state=42),
                        param_grid,cv=4,                             #Creates the function to find the best model for TreeClassifier using 4 folds
                        scoring='accuracy',n_jobs=1)                #n_jobs=-1: the number of jobs is set to the number of CPU cores
grid_search.fit(X_train,y_train)
print(grid_search.best_estimator_)
grid_search.best_estimator_.fit(X_train,y_train)                                     #Runs GridSearch
from sklearn.metrics import accuracy_score
y_pred = grid_search.best_estimator_.predict(X_test)
print(accuracy_score(y_test, y_pred))

#PART EXERCISE 08
#a Continuing the previous exercise, generate 1,000 subsets of the training
# set, each containing 100 instances selected randomly. Hint: you can use 
# Scikit-Learn’s ShuffleSplit class for this. 
from sklearn.model_selection import ShuffleSplit
#Yields indices to split data into training and test sets
mini_sets=[]
rs=ShuffleSplit(n_splits=1000,train_size=100,random_state=42)
#n_splits: numbers of re-shuffling and splittings
#train_size
for mini_train_index, mini_test_index in rs.split(X_train):
    X_mini_train = X_train[mini_train_index]
    y_mini_train = y_train[mini_train_index]
    mini_sets.append((X_mini_train, y_mini_train))

#b Train one Decision Tree on each subset, using the best hyperparameter 
# values found above. Evaluate these 1,000 Decision Trees on the test set.
# Since they were trained on smaller sets, these Decision Trees will 
# likely perform worse than the first Decision Tree, achieving only about 
# 80% accuracy.
from sklearn.base import clone
forest=[clone(grid_search.best_estimator_) for _ in range(1000)]
#forest is a list of 1000 models, all the same best_estimator_
accuracy_scores=[]

for tree, (X_mini_train, y_mini_train) in zip(forest, mini_sets):
    tree.fit(X_mini_train, y_mini_train)                         #Train each model with the mini_sets   
    y_pred = tree.predict(X_test)                                #Predict using the whole train_set
    accuracy_scores.append(accuracy_score(y_test, y_pred))       #Accuracy measurement for each model

import numpy as np
print(np.mean(accuracy_scores))

#c Now comes the magic. For each test set instance, generate the 
# predictions of the 1,000 Decision Trees, and keep only the most 
# frequent prediction (you can use SciPy’s mode() function for this). 
# This gives you majority-vote predictions over the test set
Y_pred = np.empty([1000, len(X_test)], dtype=np.uint8)  #Creates the empty matrix to receive the predictions

for tree_index,tree in enumerate(forest):               #For each tree model
    Y_pred[tree_index]=tree.predict(X_test)             #Each model predict the output for the test_set

from scipy.stats import mode
y_pred_majority_votes,n_votes=mode(Y_pred, axis=0)      #Selects the most predicted output

#d Evaluate these predictions on the test set: you should obtain a 
# slightly higher accuracy than your first model (about 0.5 to 1.5% 
# higher).
print(accuracy_score(y_test, y_pred_majority_votes.reshape([-1]))) 

end = time.time()
print(end - start)