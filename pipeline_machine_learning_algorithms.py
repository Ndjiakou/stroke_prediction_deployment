# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 08:00:11 2022

@author: HP
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
#from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTENC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
#from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler, PowerTransformer, MaxAbsScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error
##DATA PREPARATION 
stroke_data = pd.read_csv("C:\\Users\\HP\\Desktop\\stroke prediction\\healthcare-dataset-stroke-data.csv")

#print number of null value in the whole dataset
stroke_data.isna().sum().sum()
#number of missing values in the data
stroke_data.isnull().sum()
#outliers distribution, replace the null value of bmi with median
stroke_data.fillna(stroke_data["bmi"].median(), inplace=True)
# delete the id cause it is not useful
stroke_data.drop("id",inplace=True,axis=1)

# # transform text data entry into numerical form
# transform female=0 , male=1, other=2
class_mapping = {label: idx for idx, label in enumerate(np.unique(stroke_data["gender"]))}
stroke_data["gender"] = stroke_data["gender"].map(class_mapping)

# replace in married yes=1 and no=0
class_mapping = {label: idx for idx, label in enumerate(np.unique(stroke_data["ever_married"]))}
stroke_data["ever_married"] = stroke_data["ever_married"].map(class_mapping)

# replace in residence_type rural= and urban=
class_mapping = {label: idx for idx, label in enumerate(np.unique(stroke_data["Residence_type"]))}
stroke_data["Residence_type"] = stroke_data["Residence_type"].map(class_mapping)

# #preprocess the string data of work_type and smoking_status
#  convert any element of the listed columns into string
stroke_data[['work_type','smoking_status']] = stroke_data[['work_type','smoking_status']].astype(str)
#  convert categorical values from numeric
#create a news database with 5110 features , 18 colonnes
transpose = pd.get_dummies(stroke_data[['work_type','smoking_status']])
stroke_dummies_df = pd.concat([stroke_data,transpose],axis=1)[['age','hypertension','heart_disease','ever_married','Residence_type','avg_glucose_level','bmi','gender','work_type_Govt_job','work_type_Never_worked','work_type_Private','work_type_Self-employed','work_type_children','smoking_status_Unknown','smoking_status_formerly smoked','smoking_status_never smoked','smoking_status_smokes','stroke']]


# find which parameters influence the most(find the best accurancy method)
#features selection technique based importance
Z = stroke_dummies_df.iloc[:,0:17] #independent columns
W = stroke_dummies_df.iloc[:,17] #target columnstroke
model = ExtraTreesClassifier()
model.fit(Z,W)
#print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=Z.columns)
F=feat_importances.nlargest(5)


#use the important data to built the test and train data
# This is what the train/test dataset looks like if we do not apply SMOTENC.
X=stroke_dummies_df[['age','hypertension','heart_disease','ever_married','Residence_type','avg_glucose_level','bmi','gender','work_type_Govt_job','work_type_Never_worked','work_type_Private','work_type_Self-employed','work_type_children','smoking_status_Unknown','smoking_status_formerly smoked','smoking_status_never smoked','smoking_status_smokes']]
y= stroke_dummies_df['stroke']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)

#handling oversampling
smote=SMOTE()
train_x,train_y=smote.fit_resample(X_train,y_train)
test_x,test_y=smote.fit_resample(X_test, y_test)


#build the pipelines
new_pipe=make_pipeline(StandardScaler(),
PCA(n_components=17),
RandomForestClassifier(criterion="entropy",max_depth=10,min_samples_leaf=5,min_samples_split=2))
new_pipe.fit(train_x, train_y)
y_pred = new_pipe.predict(test_x)
print('Test accuracy random classifier: %.4f' % accuracy_score(y_pred, test_y))
print('Mean Absolute Error: ', mean_absolute_error(y_pred, test_y))
# # finding the score using a cross technique on pipeline rdc
# score_pipe=cross_val_score(pipe,train_x, train_y,scoring='average_precision',cv=10)
# print(score_pipe)Test accuracy random classifier: 0.9336
#Mean Absolute Error:  0.06635802469135803
# print("Avg train random classifier :",np.average(score_pipe))
# # finding the score using a cross technique for test sample
# score_pipe_test=cross_val_score(pipe,test_x, test_y,scoring='average_precision',cv=10)
# print(score_pipe_test)
# print("Avg test random classifier :",np.average(score_pipe_test))

# #analyze the results of the random forest pipeline
labels_tree = np.unique(y_pred)
confusion_mat = confusion_matrix(test_y, y_pred, labels=labels_tree)
accuracy_tree = (y_pred == test_y).mean()
print("Labels:", labels_tree)
print("Confusion Matrix random forest:\n", confusion_mat)

# #create a pkl model file
# pickle.dump(pipe,open('pipe.pkl','wb'))

## create a new pipe withour the standard scaler and  save it 
#new_pipe=StandardScaler(), RandomForestClassifier(criterion="entropy",max_depth=10,min_samples_leaf=5,min_samples_split=2)

## create a pkl model file and save it
pickle.dump(new_pipe,open('new_pipe.pkl','wb'))