# -*- coding: utf-8 -*-
"""
Early Exploration - Logistic Regression Baseline
==================================================
Initial exploration of credit card credit risk scoring using Logistic
Regression with SMOTE oversampling. This was the first iteration of the
model, using basic credit card features (salary, balance, transaction
amount, demographics).

Includes:
  - RFE (Recursive Feature Elimination) for feature selection
  - Statsmodels Logit for statistical significance testing
  - SMOTE for class balancing
  - ROC curve and classification report evaluation

This file represents the starting point before the model evolved into
the multi-algorithm comparison and MLP-based production system.

@author: sharjeel.jalil
"""

# Logistic Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE


df = pd.read_csv('FinalData.txt',sep='\t', converters = {'acctno':str})

df.drop('acctno', axis=1,inplace=True)
df.drop('limit', axis=1,inplace=True)
df.drop('cust_class', axis=1,inplace=True)
df.drop('O_Code', axis=1,inplace=True)
df.drop('marital_status', axis=1,inplace=True)
9df.drop('Category', axis=1,inplace=True)
df.drop('sex', axis=1,inplace=True)
#df.drop('Defaulter', axis=1,inplace=True)
df.drop('Defaulter_Code', axis=1,inplace=True)
df.drop('ID_NUMBER', axis=1,inplace=True)
df.drop('Debit_Credit_Cd', axis=1,inplace=True)
#df.drop('age', axis=1,inplace=True)
#df.drop('Marital_Status_Code', axis=1,inplace=True)
#df.drop('Category_Code', axis=1,inplace=True)
#df.drop('Gender_Code', axis=1,inplace=True)


df['Avg_Bal'] = df.Avg_Bal.astype(int)
df['Total_Transaction_Amt'] = df.Total_Transaction_Amt.astype(int)
df2 = df.dropna()
df.info()


count_no_default = len(df[df['Defaulter']=='No'])
count_no_default
count_default = len(df[df['Defaulter']=='Yes'])
count_default
pct_of_no_default = count_no_default/(count_no_default + count_default)
print("percentage of no default is", pct_of_no_default*100)
pct_of_default = count_default/(count_no_default + count_default)
print("percentage of default", pct_of_default*100)


#le = preprocessing.LabelEncoder()
#df2['sex']=le.fit_transform(df2['sex'])
#df2['Category']=le.fit_transform(df2['Category'])
#df2['marital_status']=le.fit_transform(df2['marital_status'])


X = df2.loc[:, df2.columns != 'Defaulter']
y = df2['Defaulter']

#####

from imblearn.datasets import fetch_datasets
from kmeans_smote import KMeansSMOTE

[print('Class {} has {} instances'.format(label, count))
 for label, count in zip(*np.unique(y, return_counts=True))]

kmeans_smote = KMeansSMOTE(
    kmeans_args={
        'n_clusters': 100
    },
    smote_args={
        'k_neighbors': 10
    }
)
X_resampled, y_resampled = kmeans_smote.fit_sample(X, y)

[print('Class {} has {} instances after oversampling'.format(label, count))
 for label, count in zip(*np.unique(y_resampled, return_counts=True))]
#########

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


#Over-Sampling
os = SMOTE(random_state=3)
columns = X_train.columns
os_data_X,os_data_y=os.fit_sample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['y'])
print("length of oversampled data is ",len(os_data_X))
print("Number of no defaulter in oversampled data",len(os_data_y[os_data_y['y']=='No']))
print("Number of defaulter",len(os_data_y[os_data_y['y']=='Yes']))
print("Proportion of no defaulter data in oversampled data is ",len(os_data_y[os_data_y['y']=='No'])/len(os_data_X))
print("Proportion of defaulter data in oversampled data is ",len(os_data_y[os_data_y['y']=='Yes'])/len(os_data_X))


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(os_data_X)
X_test = sc.transform(os_data_y.values)


# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
rfe = RFE(logreg, 20)
rfe = rfe.fit(os_data_X, os_data_y.values.ravel())
print(rfe.support_)
print(rfe.ranking_)

cols=["mth_basic_salary", "Avg_Bal", "Total_Transaction_Amt", "Marital_Status_Code", "Category_Code", "Gender_Code", "age"] 

X=os_data_X[cols]
y=os_data_y['y']

import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)


y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
print(classification_report(y_test, y_pred, digits = 4))



from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

X=os_data_X[cols]
y=os_data_y['y']
classifier = LogisticRegression(random_state = 0)
classifier.fit(, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
cm = confusion_matrix(y_test, y_pred)
cm
print(classification_report(y_test, y_pred, digits = 4))

probs = classifier.predict_proba(X_test)
probs = probs[:, 1]

##Computing false and true positive rates
fpr, tpr,_=roc_curve(classifier.predict(X_train),y_train,drop_intermediate=False)

import matplotlib.pyplot as plt
plt.figure()
##Adding the ROC
plt.plot(fpr, tpr, color='red',
 lw=2, label='ROC curve')
##Random FPR and TPR
plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
##Title and label
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve')
plt.show()



# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
cm
print(classification_report(y_test, y_pred, digits = 4))

auc = roc_auc_score(y_test, probs)
print('AUC: %.3f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, probs)