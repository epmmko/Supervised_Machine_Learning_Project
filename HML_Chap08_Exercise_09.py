#9.	Load the MNIST dataset (introduced in Chapter 3) and split it into a training set and a test set (take the first 60,000 
#instances for training, and the remaining 10,000 for testing). Train a Random Forest classifier on the dataset and time how long it 
#takes, then evaluate the resulting model on the test set. Next, use PCA to reduce the dataset’s dimensionality, with an explained 
#variance ratio of 95%. Train a new Random Forest classifier on the reduced dataset and see how long it takes. Was training much 
#faster? Next evaluate the classifier on the test set: how does it compare to the previous classifier?
import time
start = time.time()

from sklearn.datasets import fetch_mldata
mnist=fetch_mldata('MNIST original')       #70,000 numbers between 0-9

X,y=mnist['data'],mnist['target']          #Rows are intances, Columns are features (784 features=28x28pixeles). Each pixel (0:white -> 255:black)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=60000,random_state=42)                                  

from sklearn.ensemble import RandomForestClassifier
rnd_clf=RandomForestClassifier(random_state=42)                                #Three Model Classifiers 
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
reduced_clf=Pipeline([
        ('pca',PCA(n_components=0.95)),
        ('rnd_reduced_clf',RandomForestClassifier(random_state=42))
        ])
from sklearn.metrics import accuracy_score

for clf in (rnd_clf,reduced_clf):                                              #Final performance on the test set
    start = time.time()
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    print(clf.__class__.__name__,accuracy_score(y_test,y_pred))                #Accuracy is improved when the classifiers are grouped with voting
    end = time.time()
    print(end - start)

#RandomForestClassifier Score:0.9482 Time:3.5404531955718994
#Pipeline               Score:0.8919 Time:13.349834442138672