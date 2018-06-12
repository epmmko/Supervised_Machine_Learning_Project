import os
import tarfile
from six.moves import urllib

Download_Root="https://raw.githubusercontent.com/ageron/handson-ml/master/"
Housing_Path=os.path.join("datasets","housing") # Join the 2 path components.The return value is the concatenation of "datasets" and any member of "housing"
Housing_URL=Download_Root+"datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=Housing_URL,housing_path=Housing_Path):
    if not os.path.isdir(housing_path): #If housing_path is not an existing directory, then:
        os.makedirs(housing_path)       #The housing_path directory is created.
    tgz_path=os.path.join(housing_path,"housing.tgz")
    urllib.request.urlretrieve(housing_url,tgz_path) #Copy the object in "housing_url" to a local file located at "tgz_path"
    housing_tgz=tarfile.open(tgz_path)               #Return the TARfile from the path "tgz_path"
    housing_tgz.extractall(path=housing_path)        #Extract all members from housing_tgz to the path "housing_path"
    housing_tgz.close()                              #Close the TARfile housing_tgz

import pandas as pd #Using "pd" as alias for "pandas"

def load_housing_data(housing_path=Housing_Path):     #Function to load the data using Pandas
    csv_path=os.path.join(housing_path,"housing.csv") 
    return pd.read_csv(csv_path)

housing=load_housing_data()   #Save the data in the DataFrame variable "housing"
housing.info() #Give us general information about the DataFrame "housing"
housing["ocean_proximity"].value_counts() #List of categories for the attribute "ocean_proximity" and the frecuency of each object
housing.describe()  #Shows a summary of the numerical attributes

import matplotlib.pyplot as plt             #Uses plt as alias of matplotlib.pyplot
housing.hist(bins=50,figsize=(20,15))       #Creation of histogram for each attribute. 50 classes. 20in width - 15in height
plt.show()                                  #Plot the graph

import numpy as np                                      #Use np as alias of numpy
def split_train_test(data,test_ratio):                  #Function to split the data in train and test sets. test_ratio indicate the amount of test data 1 is 100%
    shuffled_indices=np.random.permutation(len(data))   #Randomly permute the numbers (0->len(data))
    test_set_size=int(len(data)*test_ratio)             #Number of instances for test set
    test_indices=shuffled_indices[:test_set_size]       #Instances selected for test set(The "first test_set_size_th")
    train_indices=shuffled_indices[test_set_size:]      #Instances selected for train set(The remaining instances)
    return data.iloc[train_indices],data.iloc[test_indices] #.iloc uses the index to select all the row

train_set,test_set=split_train_test(housing,0.2)       #assign the instances for trainning and testing
print(len(train_set),"train +",len(test_set),"test")

from sklearn.model_selection import train_test_split                       #Now we use a built function from Scikit-Learn package

train_set,test_set=train_test_split(housing,test_size=0.2,random_state=42) #random_state assures that we will get the same split everytime
print(len(train_set),"train +",len(test_set),"test")

#The median income is an important attribute, so the split proccedure should take in count the distribution of the median income
housing["income_cat"]=np.ceil(housing["median_income"]/1.5)             #round to the next integer
housing["income_cat"].where(housing["income_cat"]<5,5.0,inplace=True)   #If the condition is satisfied, housing[] kepts the values, if not the values are replaced by 5

from sklearn.model_selection import StratifiedShuffleSplit                  #Now we use a built function from Scikit-Learn package

split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)      #It has the information for the splitting procedure
for train_index,test_index in split.split(housing,housing["income_cat"]):   #For each income category, train and text indexes are selected
    strat_train_set=housing.loc[train_index]                                #Train set selection
    strat_test_set=housing.loc[test_index]                                  #Test set selection

#Elimination of the column "income_cat"
for set_ in (strat_train_set,strat_test_set):      #For each set, do:
    set_.drop("income_cat",axis=1,inplace=True)    #Drop "income_cat".axis=1 refers to column
    
#Making a copy of the train_set
housing=strat_train_set.copy()

#Scatterplot
housing.plot(kind="scatter",x="longitude",y="latitude",alpha=0.4,
            s=housing["population"]/100,label="population",figsize=(20,14),
            c="median_house_value",cmap=plt.get_cmap("jet"),colorbar=True,)
plt.legend()
#alpha parameter degrade the intensity of the color in the less populated zones
#s option uses the radius of circle to represent the population
#c option uses the color distribution to represent the median_house_value

#Correlation between attributes
corr_matrix=housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)  #Show the correlation index between "median_house_value" and the rest of attributes in a descending order

#Scattering plot between attributes with highest correlation index
from pandas.plotting import scatter_matrix
attributes=["median_house_value","median_income","total_rooms","housing_median_age"]
scatter_matrix(housing[attributes],figsize=(12,8))

housing.plot(kind="scatter",x="median_income",y="median_house_value",alpha=0.1)

#Combination of attributes to use more useful variables
housing["rooms_per_household"]=housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"]=housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]

#Correlation between attributes
corr_matrix=housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)  #Show the correlation index between "median_house_value" and the rest of attributes in a descending order

#PREPARING THE DATA FOR MACHINE LEARNING ALGORITHMS
housing=strat_train_set.drop("median_house_value",axis=1)
housing_labels=strat_train_set["median_house_value"].copy()

#Taking care of missing data
from sklearn.preprocessing import Imputer
imputer=Imputer(strategy="median")                   #We want to replace each attribute's missing values with the median of that attribute
housing_num=housing.drop("ocean_proximity",axis=1)   #Copy withouth the text attribute
imputer.fit(housing_num)                             #
imputer.statistics_                                  #Shows the results (median for each attribute)
X=imputer.transform(housing_num)                     #Transform the training set by replacing the missing values by the learned values
housing_tr=pd.DataFrame(X,columns=housing_num.columns)#Convert the array to a DataFrame

#Handling Text and Categorical Attributes
housing_cat=housing["ocean_proximity"]                      #Categorical attribute. It is a serie
housing_cat=pd.DataFrame(data=housing_cat)                  #DataFrame converts from serie --> DataFrame (includes the index)

from future_encoders import OneHotEncoder                   #This function assign 1 for when the instance has that category
cat_encoder=OneHotEncoder(sparse=False)
housing_cat_1hot=cat_encoder.fit_transform(housing_cat)     #Then, create a sparse matrix with the location of nonzeros.
housing_cat_1hot
cat_encoder.categories_


#Custom Transformer
from sklearn.base import BaseEstimator,TransformerMixin

rooms_ix,bedrooms_ix,population_ix,household_ix=3,4,5,6                                    #Columns (location) of each attribute

class CombinedAttributesAdder(BaseEstimator,TransformerMixin):                             #The New class is using two base classes
    def __init__(self,add_bedrooms_per_room=True):                                           #_init_ is the constructor for the class
        self.add_bedrooms_per_room=add_bedrooms_per_room
    def fit(self,X,y=None):
        return self                                                                        #Nothing else to do
    def transform(self,X,y=None):                                                          #This function does the combination of parameters as it was done before
        rooms_per_household=X[:,rooms_ix]/X[:,household_ix]
        population_per_household=X[:,population_ix]/X[:,household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room=X[:,bedrooms_ix]/X[:,rooms_ix]
            return np.c_[X,rooms_per_household,population_per_household,bedrooms_per_room]
        else:
            return np.c_[X,rooms_per_household,population_per_household]
        
attr_adder=CombinedAttributesAdder(add_bedrooms_per_room=False)    #Decide not to use the bedrooms_per_room attribute
housing_extra_attribs=attr_adder.transform(housing.values)         #The new attributes are added
print(housing_extra_attribs)


#Transformation using Pipelines from Scikit-Learn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline=Pipeline([('imputer',Imputer(strategy="median")),      #Uses the median of the numerical attribute for the missing values
                      ('attribs_adder',CombinedAttributesAdder()),  #Adds new combined attributes
                      ('std_scaler',StandardScaler()),              #Scales the values usign standarization
                      ])

housing_num_tr=num_pipeline.fit_transform(housing_num)              #Actually performs the transformation

#Feeding a Pandas DataFrame containing non-numerical column
class DataFrameSelector(BaseEstimator,TransformerMixin): #The New class is using two base classes
    def __init__(self,attribute_names):                  #_init_ is the constructor for the class
        self.attribute_names=attribute_names
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        return X[self.attribute_names].values            #Transforms the dataframe according

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

num_attribs=list(housing_num)
cat_attribs=['ocean_proximity']

num_pipeline=Pipeline([('selector',DataFrameSelector(num_attribs)), #Selects the numerical attributes
                      ('imputer',Imputer(strategy="median")),       #Uses the median of the numerical attribute for the missing values
                      ('attribs_adder',CombinedAttributesAdder()),  #Adds new combined attributes
                      ('std_scaler',StandardScaler()),              #Scales the values usign standarization
                      ])

cat_pipeline=Pipeline([('selector',DataFrameSelector(cat_attribs)), #Selects the categorical attributes
                      ('cat_encoder',OneHotEncoder(sparse=False)),  #Creates a matrix with 1 and 0 for categorical attributes
                      ])

from sklearn.pipeline import FeatureUnion                                   #This class create a pipeline with both the numerical and categorical pipeline

full_pipeline=FeatureUnion(transformer_list=[('num_pipeline',num_pipeline), 
                                            ('cat_pipeline',cat_pipeline),
                                            ])   

housing_prepared=full_pipeline.fit_transform(housing)                       #This performs the whole data preparation

print(housing_prepared)

#TRAINING USING LINEAR REGRESSION
from sklearn.linear_model import LinearRegression

lin_reg=LinearRegression()
lin_reg.fit(housing_prepared,housing_labels)   #housing_prepared contains the 16 attributes, Housing_labels is y.

#Evaluation
some_data=housing.iloc[:5]                               #Selects the first 5 rows
some_labels=housing_labels.iloc[:5]                      #Reference values
some_data_prepared=full_pipeline.transform(some_data)    #Execute the whole pipeline using the sample data
print("Predictions:",lin_reg.predict(some_data_prepared))

#Measurement of the RMSE
from sklearn.metrics import mean_squared_error
housing_predictions=lin_reg.predict(housing_prepared)
lin_mse=mean_squared_error(housing_labels,housing_predictions)
lin_rmse=np.sqrt(lin_mse)
lin_rmse

#NonLinear Regression
from sklearn.tree import DecisionTreeRegressor                  #This model is capable of finding complex nonlinear relationship in the data

tree_reg=DecisionTreeRegressor()
tree_reg.fit(housing_prepared,housing_labels)                   #X=housing_prepared,Y=housing_labels
housing_predictions=tree_reg.predict(housing_prepared)          #Calculates the prediction
tree_mse=mean_squared_error(housing_labels,housing_predictions) #X=real_values,Y=estimation
tree_rmse=np.sqrt(tree_mse)
tree_rmse

#Evaluation using Cross-Validation
from sklearn.model_selection import cross_val_score
#It randomly splits the training set into 10 distinct subsets (folds), then pick a different fold each time and trains it on 
#the other 9 folds.


scores=cross_val_score(tree_reg,housing_prepared,housing_labels, #Evaluation the DecisionTree function
                      scoring="neg_mean_squared_error",cv=10)
tree_rmse_scores=np.sqrt(-scores)                                #A utility function is expected (higher value is better). Scores are the opposite of the MSE

def display_scores(scores):
    print("Scores:",scores)
    print("Mean:",scores.mean())
    print('Standard Deviation:',scores.std())

display_scores(tree_rmse_scores)

lin_scores=cross_val_score(lin_reg,housing_prepared,housing_labels,  #Evaluation the LinearRegression model
                          scoring="neg_mean_squared_error",cv=10)    #10 Folds
lin_rmse_scores=np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)

#Random Forest Model Evaluation
from sklearn.ensemble import RandomForestRegressor  #Works by training many DecisionTress on random subsets, then averaging the predictions
forest_reg=RandomForestRegressor()
forest_reg.fit(housing_prepared,housing_labels)

forest_scores=cross_val_score(forest_reg,housing_prepared,housing_labels,
                             scoring="neg_mean_squared_error",cv=10)
forest_rmse_scores=np.sqrt(-forest_scores)
forest_rmse_scores
display_scores(forest_rmse_scores)

#FINE_TUNE THE MODEL
from sklearn.model_selection import GridSearchCV #Finds the best combination of hyperparameters using cross-vaidation
#In this example RandomForest is Used
param_grid=[{'n_estimators':[3,10,30],'max_features':[2,4,6,8]},              #Run RandomForest for each combination of n_estimator and max_feature
           {'bootstrap':[False],'n_estimators':[3,10],'max_features':[2,3,4]},
           ]
grid_search=GridSearchCV(forest_reg,param_grid,cv=5,                         #Creates the function to find the best model for RandomForest using 5 folds
                        scoring='neg_mean_squared_error')

grid_search.fit(housing_prepared,housing_labels)                             #Runs GridSearch

grid_search.best_params_                    #Find the best combination of parameters for RandomForestRegressor

#Evaluation of scores
cvres=grid_search.cv_results_
for mean_score,params in zip(cvres['mean_test_score'],cvres['params']):
    print(np.sqrt(-mean_score),params)
    
#Best models and their error
feature_importances=grid_search.best_estimator_.feature_importances_                #Save the information about the importances of each feature in the case of the best combination of estimators
extra_attribs=['rooms_per_household','population_per_household','bedrooms_per_room']
cat_encoder=cat_pipeline.named_steps['cat_encoder']
cat_one_hot_attribs=list(cat_encoder.categories_[0])
attributes=num_attribs+extra_attribs+cat_one_hot_attribs                            #The whole list of attributes used by the model
sorted(zip(feature_importances,attributes),reverse=True)                           #Sorts in a descent order the attributes importance

#Evaluation using the Test Set
final_model=grid_search.best_estimator_
x_test=strat_test_set.drop('median_house_value',axis=1)
y_test=strat_test_set['median_house_value'].copy()

x_test_prepared=full_pipeline.transform(x_test)

final_predictions=final_model.predict(x_test_prepared)
final_mse=mean_squared_error(y_test,final_predictions)
final_rmse=np.sqrt(final_mse)
final_rmse

#EXERCISE 01
from sklearn.svm import SVR  #Support Vector Machine Regressor
#In this exercise SVR is Used
SVR_reg=SVR()
param_grid=[{"kernel":["linear"],"C":[1,2,4]},                   #Run SRV for each combination of kernel and C(penalty parameter of the error)
           {"kernel":["rbf"],"C":[1,2,4]},
           ]
grid_search=GridSearchCV(SVR_reg,param_grid,cv=5,          #Creates the function to find the best model for SVR using 5 folds
                        scoring='neg_mean_squared_error')

grid_search.fit(housing_prepared,housing_labels)           #Runs GridSearch

#Evaluation of scores
cvres=grid_search.cv_results_
for mean_score,params in zip(cvres['mean_test_score'],cvres['params']):
    print(np.sqrt(-mean_score),params)
    
#EXERCISE 02
#In this exercise SVR is Used
from sklearn.model_selection import RandomizedSearchCV

param_grid={"kernel":["linear","rbf"],"C":[1,2,4,8,16]}      #Run SRV for each combination of kernel and C(penalty parameter of the error)

random_search=RandomizedSearchCV(SVR_reg,param_grid,cv=5,    #Creates the function to find the best model for SVR using 5 folds
                        scoring='neg_mean_squared_error')

random_search.fit(housing_prepared,housing_labels)           #Runs RandomSearch

#Evaluation of scores
cvres=random_search.cv_results_
for mean_score,params in zip(cvres['mean_test_score'],cvres['params']):
    print(np.sqrt(-mean_score),params)
    
#EXERCISE 03
from sklearn.base import BaseEstimator,TransformerMixin

def indices_of_top_k(arr,k):
    return np.sort(np.argpartition(np.array(arr),-k)[-k:])
#numpy.array creates an array using the object "arr"
#numpy.argpartition returns a partition of array of indexes in an sorted position
#numpy.sort returns a sorted copy of an array, [-k:] this change the order.

class TopFeatureSelector(BaseEstimator,TransformerMixin):
    def __init__(self, feature_importances, k):
        self.feature_importances=feature_importances
        self.k=k
    def fit(self, X, y=None):
        self.feature_indices_=indices_of_top_k(self.feature_importances,self.k)
        return self
    def transform(self, X):
        return X[:,self.feature_indices_]
    
k=5                                                           #Number of features to consider
top_k_feature_indices=indices_of_top_k(feature_importances,k) 
top_k_feature_indices
np.array(attributes)[top_k_feature_indices]

preparation_and_feature_selection_pipeline=Pipeline([
    ('preparation',full_pipeline),
    ('feature_selection',TopFeatureSelector(feature_importances,k))
])

housing_prepared_top_k_features=preparation_and_feature_selection_pipeline.fit_transform(housing)
housing_prepared_top_k_features[0:3]

#EXERCISE 04
prepare_select_and_predict_pipeline=Pipeline([
    ('preparation',full_pipeline),
    ('feature_selection',TopFeatureSelector(feature_importances,k)),
    ('svm_reg',SVR(**random_search.best_params))
])

prepare_select_and_predict_pipeline.fit(housing, housing_labels)