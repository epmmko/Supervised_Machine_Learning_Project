import time
start = time.time()

#a Generate a moons dataset using make_moons(n_samples=10000, noise=0.4). 
from sklearn.datasets import make_moons
X,y=make_moons(n_samples=10000, noise=0.4, random_state=53)

#b Split it into a training set and a test set using train_test_split().
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

#c Use grid search with cross-validation (with the help of the GridSearchCV
# class) to find good hyperparameter values for a DecisionTreeClassifier. 
#Hint: try various values for max_leaf_nodes. 
from sklearn.tree import DecisionTreeClassifier
param_grid={'max_depth':[3,5,7,10,15],'min_samples_leaf':[1,5,10]} #Run DecisionTreeClassifier for each combination of max_depth and min_sample_leaf

from sklearn.model_selection import GridSearchCV                     #Finds the best combination of hyperparameters using cross-vaidation
grid_search=GridSearchCV(DecisionTreeClassifier(random_state=42),
                        param_grid,cv=4,                             #Creates the function to find the best model for TreeClassifier using 4 folds
                        scoring='accuracy',n_jobs=1)                #n_jobs=-1: the number of jobs is set to the number of CPU cores
grid_search.fit(X_train,y_train)
print(grid_search.best_estimator_)

#d Train it on the full training set using these hyperparameters and 
#measure your model’s performance on the test set. You should get roughly 
#85% to 87% accuracy.
grid_search.best_estimator_.fit(X_train,y_train)                                     #Runs GridSearch

from sklearn.metrics import accuracy_score
y_pred = grid_search.best_estimator_.predict(X_test)
print(accuracy_score(y_test, y_pred))

from  sklearn.tree  import export_graphviz
export_graphviz(                             #exports the image
        grid_search.best_estimator_,         #Classifier
        out_file='make_moon.dot',            #File name
        rounded=True,                        #Draws node boxes with rounded corners and use Helvetica fonts instead of Times-Roman
        filled=True                          #Paints nodes to indicate majority class for classification
    )
import os
import pydot
os.environ["PATH"] += os.pathsep + r'C:\Users\victo\Anaconda3\envs\datascience\Library\bin\graphviz' #Solves problem with the dot.exe path
(graph,)=pydot.graph_from_dot_file('make_moon.dot')
graph.write_png('make_moon.png')             #Converts dot file to png

end = time.time()
print(end - start)