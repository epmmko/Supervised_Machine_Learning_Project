from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
mnist

X,y=mnist['data'],mnist['target']
X.shape                           #Rows are intances, Columns are features (784 features=28x28pixeles). Each pixel (0:white -> 255:black)
y.shape                           #70,000 numbers between 0-9

import matplotlib
import matplotlib.pyplot as plt                        #Uses plt as alias for matplotlib.pyplot

some_digit=X[36000]                                    #Select the instance 36000
some_digit_image=some_digit.reshape(28,28)             #Reshape the vector to an array 28x28. The values have the information about the intesity of the color
plt.imshow(some_digit_image,cmap=matplotlib.cm.binary, #Display the image
          interpolation='nearest')
plt.axis('off')                                        #Do not show the axis
plt.show()                                             #Only shows the image

#Splitting in Train and Test Set
X_train,X_test,y_train,y_test=X[:60000],X[60000:],y[:60000],y[60000:]
#The first 60,000 instances are for trainning, the remaining is for testing

import numpy as np                                              #Uses np as alias of numpy
shuffle_index=np.random.permutation(60000)                      #Perform a random permutation of the number 0-60,000
X_train,y_train=X_train[shuffle_index],y_train[shuffle_index]   #Shuffles the train_set to guarantee that all cross-validation fikds will be similar

#TRAINING A BINARY CLASSIFIER
#Example of a 5-classifier
y_train_5=(y_train==5) #True for all 5s, False for other digits.
y_test_5=(y_test==5)

from sklearn.linear_model import SGDClassifier
#Stochastic Gradient Descent classifier deals with training instances independetly, one at a time.

sgd_clf=SGDClassifier(random_state=42)  #Create the classifier function, the 42 let us to obtain the same result everytime.
sgd_clf.fit(X_train,y_train_5)          #fits a model using the train_set and the boolean desired result
sgd_clf.predict([some_digit])           #Evaluates the row[36,000]

#PERFORMANCE MEASURES
#Using Cross-Validation
from sklearn.model_selection import cross_val_score                #Evaluates the model
cross_val_score(sgd_clf,X_train,y_train_5,cv=3,scoring='accuracy')
#sgd_clf has already the model to predict.
#cv is the number of fold
#'accuracy' is the ratio of goog predictions/total predictions

#Using a custom classifier for "not-5" images
from sklearn.base import BaseEstimator

class Never5Classifier(BaseEstimator):
    def fit(self,X,y=None):
        pass                                        #Nothing happends, but a syntax is required
    def predict(self,X):
        return np.zeros((len(X),1),dtype=bool)      #Classifies everything as non-5
    #len() returns the length of X
    #np.zeros((rows,columns),boolean type)

never_5_clf=Never5Classifier()
cross_val_score(never_5_clf,X_train,y_train_5,cv=3,scoring='accuracy') #This has a 90% percent of accuracy, but it is a whorthless function

#Evaluation using Confusion Matrix
from sklearn.model_selection import cross_val_predict
y_train_pred=cross_val_predict(sgd_clf,X_train,y_train_5,cv=3)
#Uses the sgd_clf classifier, and 3 fold

from sklearn.metrics import confusion_matrix
confusion_matrix(y_train_5,y_train_pred)      #Compare the train_set results with the prediction
#         Not-5 Predicted | 5 Predicted
# Not-5
#   5 

#Precision=TP/(TP+FP)
#Recall=TP/(TP+FN)
#F1_Score=2/(1/precision+1/recall)
from sklearn.metrics import precision_score,recall_score,f1_score

precision_score(y_train_5,y_train_pred)
recall_score(y_train_5,y_train_pred)
f1_score(y_train_5,y_train_pred)

y_scores=cross_val_predict(sgd_clf,X_train,y_train_5,cv=3,method='decision_function')    #Gets the score from all data
#Manually we can set a threshold

from sklearn.metrics import precision_recall_curve                            
precisions,recalls,thresholds=precision_recall_curve(y_train_5,y_scores)  #Returns the corresponding precision and recall for each threshold

def plot_precision_recall_vs_threshold(precisions,recalls,thresholds):    #Creates a plotting function
    plt.plot(thresholds,precisions[:-1],'b--',label='Precision')
    plt.plot(thresholds,recalls[:-1],'g-',label='Recall')
    plt.xlabel('Threshold')                                               #Label for x axis
    plt.legend(loc='center left')                                         #location for the legend
    plt.ylim([0,1])                                                       #Range for y axis

plot_precision_recall_vs_threshold(precisions,recalls,thresholds)
plt.show()

#ROC Curve
#Receiver Operating Characterictic Curve
#True Positive Rate vs False Positive Rate
#True Positive Rate = Recall
#False Positive = FP/(FP+TN)
from sklearn.metrics import roc_curve
fpr,tpr,thresholds=roc_curve(y_train_5,y_scores)

def plot_roc_curve(fpr,tpr,label=None):
    plt.plot(fpr,tpr,linewidth=2,label=label)
    plt.plot([0,1],[0,1],'k--')
    plt.axis([0,1,0,1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

plot_roc_curve(fpr,tpr)  #The dotted line represents a purely random classifier. A good classifier stay as far away from it.
plt.show()

from sklearn.metrics import roc_auc_score
roc_auc_score(y_train_5,y_scores) #Calculates the Area Under the ROC Curve. A good classifier has a score close to 1

#ROC Curve using a Random Classifier
from sklearn.ensemble import RandomForestClassifier

forest_clf=RandomForestClassifier(random_state=42)
y_probas_forest=cross_val_predict(forest_clf,X_train,y_train_5,cv=3,method='predict_proba') #Returns an array containing a row per instance and a column per clas.
#Each class contain the probability that the given instance belongs to the given class.

y_scores_forest=y_probas_forest[:,1]                                          #score = proba of positive class
fpr_forest,tpr_forest,thresholds_forest=roc_curve(y_train_5,y_scores_forest)  #Evaluates the performance of the RandomClassifier for different thresholds
plt.plot(fpr,tpr,'b:',label='SGD')                                            #Plots the SGD performance
plot_roc_curve(fpr_forest,tpr_forest,'Random_Forest')                         #Plots the RandomForest Performance
plt.legend(loc='lower right')

#MULTICLASS CLASSIFICATION
#One-Versus-All (OvA)
#Uses a binary classifier for each digit (10), and uses the score from each one to decide the real number.
sgd_clf.fit(X_train,y_train)                                #y_train contains the numbers targeted
sgd_clf.predict([some_digit])                               #Predict the class for [36,000] instance
some_digit_scores=sgd_clf.decision_function([some_digit])   #Gets the score for each class
some_digit_scores
np.argmax(some_digit_scores)                                #Gets the maximun argumen location
sgd_clf.classes_                                            #Classes

#One vs One (OvO) strategy
#Trains a binary classifier for every pair of digit: 0 vs 1, 0 vs 2, 1 vs 2, etc.
#Get the class that wins more duels
from sklearn.multiclass import OneVsOneClassifier
ovo_clf=OneVsOneClassifier(SGDClassifier(random_state=42))
ovo_clf.fit(X_train,y_train)
ovo_clf.predict([some_digit])
len(ovo_clf.estimators_)

#Multiclassification with RandomForestClassifier
forest_clf.fit(X_train,y_train)
forest_clf.predict([some_digit])        #Predict the class directly
forest_clf.predict_proba([some_digit])  #Get the probability for each class
cross_val_score(sgd_clf,X_train,y_train,cv=3,scoring='accuracy')   #Evaluation of Multiclass using cross-validation score

#Scaling the data helps to improve the accuracy
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train.astype(np.float64))           #numpy.float64 requests using the double precision float type
cross_val_score(sgd_clf,X_train_scaled,y_train,cv=3,scoring='accuracy')   #Evaluation of Multiclass using cross-validation score

#ERROR ANALYSIS
#Confusin matrix for multiclassification
y_train_pred=cross_val_predict(sgd_clf,X_train_scaled,y_train,cv=3)
conf_mx=confusion_matrix(y_train,y_train_pred)
conf_mx
plt.matshow(conf_mx,cmap=plt.cm.gray)
plt.show()

row_sums=conf_mx.sum(axis=1,keepdims=True)    #Sums each row of the confusion matrix, gets a array (10,1)
norm_conf_mx=conf_mx/row_sums                 #Divides each cell by the corresponding total along the row
np.fill_diagonal(norm_conf_mx,0)              #Fills the diagonal with 0
plt.matshow(norm_conf_mx,cmap=plt.cm.gray)
plt.show()


#Multilabel classification system
#Classification system that outputs multiple binary labels
from sklearn.neighbors import KNeighborsClassifier   #Supports multilabel classification
y_train_large=(y_train>=7)                            #True for 7,8,9
y_train_odd=(y_train%2==1)                            #True for odd numbers
y_multilabel=np.c_[y_train_large,y_train_odd]         #Assign the 2 conditions as attribute for y_multilabel

knn_clf=KNeighborsClassifier()
knn_clf.fit(X_train,y_multilabel)                     #Trains the classfier
knn_clf.predict([some_digit])                         #Predicts for the [36,000] instance   

#Evaluation of multilabel classification system
y_train_knn_pred=cross_val_predict(knn_clf,X_train,y_multilabel,cv=3)
f1_score(y_multilabel,y_train_knn_pred,average='macro')


#EXERCISE 01
knn_clf.fit(X_train,y_train)
y_train_digit_predict=knn_clf.predict(X_train)
from sklearn.metrics import accuracy_score
accuracy_score(y_train, y_train_digit_predict)

from sklearn.model_selection import GridSearchCV                          #Finds the best combination of hyperparameters using cross-vaidation
#In this exercise KNeighborsClassifier is Used
param_grid=[{'weights':['uniform','distance'],'n_neighbors':[3,5,10]}]    #Run KNeighbors for each combination of weight and n_neighbor
           
grid_search=GridSearchCV(knn_clf,param_grid,cv=3,                         #Creates the function to find the best model for KNeighbors using 3 folds
                        scoring='accuracy',n_jobs=-1)                     #n_jobs=-1: the number of jobs is set to the number of CPU cores

grid_search.fit(X_train,y_train)                                          #Runs GridSearch
y_pred = grid_search.predict(X_test)
accuracy_score(y_test, y_pred)


#Exercise 02
from scipy.ndimage.interpolation import shift                       #Function to move the array
def shift_image(image, dx, dy):                                     #image is a vector shape with the information of the pixels
    image = image.reshape((28, 28))                                 #Reshape the vector to array (28x28)
    shifted_image = shift(image, [dy, dx], cval=0, mode="constant") #function execution. Points outside the array are filled with the "constant=0"
    return shifted_image.reshape([-1])                              #Reshape in a row with unknown size

image = X_train[1000]                                               #Image in location 10,000
shifted_image_down = shift_image(image, 0, 5)                       #Image shifted down 5 pixels
shifted_image_left = shift_image(image, -5, 0)                      #Image shifted left 5 pixels

plt.figure(figsize=(12,3))
plt.subplot(131)
plt.title("Original", fontsize=14)
plt.imshow(image.reshape(28, 28), interpolation="nearest", cmap="Greys")
plt.subplot(132)
plt.title("Shifted down", fontsize=14)
plt.imshow(shifted_image_down.reshape(28, 28), interpolation="nearest", cmap="Greys")
plt.subplot(133)
plt.title("Shifted left", fontsize=14)
plt.imshow(shifted_image_left.reshape(28, 28), interpolation="nearest", cmap="Greys")
plt.show()

X_train_augmented = [image for image in X_train]                    #Initial data, prior to be augmented
y_train_augmented = [label for label in y_train]

for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):                   #Loop to shift the images
    for image, label in zip(X_train, y_train):                      #Run in every image and label data
        X_train_augmented.append(shift_image(image, dx, dy))        #The shifted image is added to the original data collection
        y_train_augmented.append(label)                             #A copy of the label added.

X_train_augmented = np.array(X_train_augmented)                     #Convert data to array structure
y_train_augmented = np.array(y_train_augmented)

shuffle_idx = np.random.permutation(len(X_train_augmented))         #Shuffle the new augmented data randomly
X_train_augmented = X_train_augmented[shuffle_idx]
y_train_augmented = y_train_augmented[shuffle_idx]

knn_clf = KNeighborsClassifier(**grid_search.best_params_)          #Using the best parameters,the classifier is trained with the new augmented data
knn_clf.fit(X_train_augmented, y_train_augmented)
y_pred = knn_clf.predict(X_test)                                    #New prediction
accuracy_score(y_test, y_pred)                                      #Accuracy of the model