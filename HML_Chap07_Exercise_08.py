#8 Load the MNIST data (introduced in Chapter 3), and split it into a training set, a validation set, and a test set (e.g., use the
# first 40,000 instances for training, the next 10,000 for validation, and the last 10,000 for testing). Then train various classifiers,
# such as a Random Forest classifier, an Extra-Trees classifier, and an SVM. Next, try to combine them into an ensemble that outperforms
# them all on the validation set, using a soft or hard voting classifier. Once you have found one, try it on the test set. How much 
# better does it perform compared to the individual classifiers?
import time
start = time.time()

from sklearn.datasets import fetch_mldata
mnist=fetch_mldata('MNIST original')       #70,000 numbers between 0-9

X,y=mnist['data'],mnist['target']          #Rows are intances, Columns are features (784 features=28x28pixeles). Each pixel (0:white -> 255:black)

#import numpy as np
from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_squared_error

X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=50000,random_state=42)                                  
X_train,X_val,y_train,y_val=train_test_split(X_train,y_train,train_size=40000,random_state=42) 


start = time.time()
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,VotingClassifier
from sklearn.svm import SVC
                              
rnd_clf=RandomForestClassifier(random_state=42)                                #Three Model Classifiers 
extra_trees_clf=ExtraTreesClassifier(random_state=42)
svm_clf=SVC(random_state=42,probability=True)

voting_clf=VotingClassifier(                                                   #Majority rule classifier
        estimators=[('rf',rnd_clf),('et',extra_trees_clf),('svc',svm_clf)],    #List of (string,estimator) tuples
        voting='hard')                                                         #hard or soft

from sklearn.metrics import accuracy_score
for clf in (rnd_clf,extra_trees_clf,svm_clf,voting_clf):
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_val)
    print(clf.__class__.__name__,accuracy_score(y_val,y_pred))                 #Accuracy is improved when the classifiers are grouped with voting

for clf in (rnd_clf,extra_trees_clf,svm_clf,voting_clf):                       #Final performance on the test set
    y_pred=clf.predict(X_test)
    print(clf.__class__.__name__,accuracy_score(y_test,y_pred))                #Accuracy is improved when the classifiers are grouped with voting


end = time.time()
print(end - start)