# -*- coding: utf-8 -*-
"""
Feature Engineering Pipeline for Credit Risk Scoring Engine
==========================================================
Integrates data from credit card portfolio, customer demographics, deposit
accounts, and transaction history. Performs feature selection using Pearson
correlation with p-value significance testing (threshold 5%), one-hot encoding
of categorical variables, Min-Max normalization, and KMeans-SMOTE oversampling
to handle 97.5/2.5 class imbalance.

Evaluates multiple classifiers (Logistic Regression, SVM, Random Forest)
with recursive feature elimination to identify the optimal feature subset.

@author: sharjeel.jalil
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from imblearn.datasets import fetch_datasets
from kmeans_smote import KMeansSMOTE
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn import metrics
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.preprocessing import KBinsDiscretizer



#df = pd.read_csv('VCCCust360.txt',sep='\t', converters={'ID_NUMBER':str})

df1 = pd.read_excel('Cards Portfolio Jan 2019.xlsx', sheetname='Cards Portfolio Jan 2019', converters = {'cnic':str})
df2 = df1.iloc[:,[3,18]]
df2['cnic'] = [str(x).replace('-','') for x in df1['cnic']]
df2= df2.groupby(by=['cnic'], as_index=False)['ag'].max()


def Mapping(x):
    if x > 0:
        return 'Yes'
    else :
        return 'No'


df2['Defaulter'] = [Mapping(x) for x in df2["ag"]]
df2.drop('ag', axis=1,inplace=True)

df3 = pd.read_csv('VCCCustomers1.txt',sep='\t',converters = {'ID_NUMBER':str})
df3['EB_CUS_NATIONALITY'].value_counts()


df4 = pd.merge(df2,
               df3,
               left_on='cnic',
               right_on='ID_NUMBER',
               how='inner')



SelectedColumns = ['Defaulter','BankingGroup' , 'ADDRESS_TYPE', 'gender', 'ProductType' , 'relationship' , 'accountOpAge' , 'nowAge' , 'gender' , 'Avg_Bal_Month5' , 'CURRENT_ACCT' , 'SAVING_ACCT' , 'avg_deposit_bal' , 'SMS_FACILITY' , 'NO_OF_LOANS' , 'INACTIVE_CR_CARD' , 'ACTIVE_CR_CARD' , 'INTERNET_BANKING' , 'NO_OF_POLICIES' , 'CR_CARD_CUST_LIMIT' , 'sm_debit' , 'dailyATM_amt' , 'weeklyATM_amt' , 'monthlyATM_amt' , 'weeklyPOS' , 'weeklyUBP_amt' , 'monthlyUBP_amt' , 'weeklyUBP' , 'weeklyCCBLP_amt' , 'monthlyCCBLP' , 'dailyFT_amt' , 'weeklyFT_amt' , 'monthlyFT_amt' , 'weeklyFT']
dd = [x for x in df4.columns if x not in SelectedColumns]
df4.drop(dd, axis=1,inplace=True)


x = pd.get_dummies(df4[['BankingGroup', 'ProductType', 'ADDRESS_TYPE', 'gender']])
df5 = pd.concat([df4, x], sort=False, axis =1)

#df3['BankingGroupMapped'] = df3['BankingGroup'].map({'Conventional': 1 , 'IBG': 2})
#df3['ProductTypeMapped'] = df3['ProductType'].map({'C': 1 , 'S': 2 ,'R': 3 ,'O': 4})
#df3['AddressTypeMapped'] = df3['ADDRESS_TYPE'].map({'RESIDENCE': 1 , 'OFFICE': 2 ,'PERMANENT': 3 ,'TEMPORARY': 4})
#df3['BankingGroupMapped'] = df3['BankingGroup'].map({'Conventional': 1 , 'IBG': 2 ,'GOLD': 3 ,'AMEX GOLD': 4 ,'TITANIUM': 5 ,'PLATINUM': 6 })


df5.drop('BankingGroup', axis=1,inplace=True)
df5.drop('ProductType', axis=1,inplace=True)
df5.drop('ADDRESS_TYPE', axis=1,inplace=True)
df5.drop('gender', axis=1,inplace=True)


#df5['sm_debit'] = df5['sm_debit'].fillna(0)
#
#df5['dailyATM'] = df5['dailyATM'].fillna(0)
#df5['weeklyATM'] = df5['weeklyATM'].fillna(0)
#df5['monthlyATM'] = df5['monthlyATM'].fillna(0)
#
#df5['dailyPOS'] = df5['dailyPOS'].fillna(0)
#df5['weeklyPOS'] = df5['weeklyPOS'].fillna(0)
#df5['monthlyPOS'] = df5['monthlyPOS'].fillna(0)
#
#df5['dailyUBP'] = df5['dailyUBP'].fillna(0)
#df5['weeklyUBP'] = df5['weeklyUBP'].fillna(0)
#df5['monthlyUBP'] = df5['monthlyUBP'].fillna(0)
#
#df5['dailyCCBLP'] = df5['dailyCCBLP'].fillna(0)
#df5['weeklyCCBLP'] = df5['weeklyCCBLP'].fillna(0)
#df5['monthlyCCBLP'] = df5['monthlyCCBLP'].fillna(0)
#
#df5['dailyIBFT'] = df5['dailyIBFT'].fillna(0)
#df5['weeklyIBFT'] = df5['weeklyIBFT'].fillna(0)
#df5['monthlyIBFT'] = df5['monthlyIBFT'].fillna(0)
#
#df5['dailyBFT'] = df5['dailyBFT'].fillna(0)
#df5['weeklyBFT'] = df5['weeklyBFT'].fillna(0)
#df5['monthlyBFT'] = df5['monthlyBFT'].fillna(0)
#
#df5['Avg_Bal_Month1'] = df5['Avg_Bal_Month1'].fillna(0)
#df5['Avg_Bal_Month2'] = df5['Avg_Bal_Month2'].fillna(0)
#df5['Avg_Bal_Month3'] = df5['Avg_Bal_Month3'].fillna(0)
#df5['Avg_Bal_Month4'] = df5['Avg_Bal_Month4'].fillna(0)
#df5['Avg_Bal_Month5'] = df5['Avg_Bal_Month5'].fillna(0)

df5.fillna(0, inplace=True)

from sklearn.preprocessing import LabelEncoder
lb_category = LabelEncoder()
df5['Defaulter'] = lb_category.fit_transform(df5['Defaulter'])

#df5.drop('cnic', axis=1,inplace=True)
#df5.drop('ID_NUMBER', axis=1,inplace=True)
#df5.drop('ACCOUNT_NUM', axis=1,inplace=True)
#df5.drop('BRANCH_code', axis=1,inplace=True)

df5.dtypes

###########################################################################
#Data Distribution
count_no_default = len(df5[df5['Defaulter']=='No'])
count_no_default
count_default = len(df5[df5['Defaulter']=='Yes'])
count_default
pct_of_no_default = count_no_default/(count_no_default + count_default)
print("percentage of no default is", pct_of_no_default*100)
pct_of_default = count_default/(count_no_default + count_default)
print("percentage of default", pct_of_default*100)

######################################################################################

#Normalization
#normalized_df = preprocessing.normalize(df5)


x = df5.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df6 = pd.DataFrame(x_scaled, columns=df5.columns, index=df5.index)


######################################################################################

aa = preprocessing.KBinsDiscretizer(n_bins=[3, 2, 2], encode='ordinal').fit(x)



######################################################################################

#Defining Predictor and Target
X = df6.loc[:, df6.columns != 'Defaulter']
y = df6['Defaulter']


#Kmeans-Smote
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

########################################################################################

#Recursive Feature Elimination
logreg = LogisticRegression()
#svm = LinearSVC()
rfe = RFE(logreg, 40)
rfe = rfe.fit(X_resampled, y_resampled)
print(rfe.support_)
print(rfe.ranking_)


#no of features
nof_list=np.arange(1,67)            
high_score=0
#Variable to store the optimum features
nof=0           
score_list =[]

def sharjeelPower(X_train,y_train,n):
    model = LogisticRegression()
    rfe = RFE(model,nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    score = model.score(X_test_rfe,y_test)
#    score_list.append(score)
    return {
            'n': n,
            'score':score
            }

import multiprocessing as mp

pool = mp.Pool(mp.cpu_count())

processResults = []
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=0)
for n in range(len(nof_list)):
    result = pool.map(sharjeelPower,(X_train,y_train,n))
    processResults.append(result)
#    if(score>high_score):
#        high_score = score
#        nof = nof_list[n]
        
print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))


#Splitting Dataset
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=0)

######################################################################################

#Logistic Regression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
PreProb = classifier.predict_proba(X_test)[:,0]
#y_pred = np.where(PreProb > 0.5, 'No','Yes')
y_pred = classifier.predict(X_test)

print('Accuracy of classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))


######################################################################################

#Kernel SVM
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
plt.hist(y_pred)
plt.show()


#######################################################################################

#Random Forest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 1000, random_state = 42)
classifier.fit(X_train, y_train);
y_pred = classifier.predict(X_test)

estimator = classifier.estimators_[0]

from sklearn.tree import export_graphviz
import os
# Export as dot file
export_graphviz(estimator, out_file='tree.dot', 
                feature_names = X.columns,
#                class_names = y.columns,
                rounded = True, proportion = False, 
                precision = 2, filled = True)

os.system('dot -Tpng tree.dot -o tree1.png')

#######################################################################################


#Accuracy
print('Accuracy of classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))

#Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#Classification Report
print(classification_report(y_test, y_pred, digits = 4))

######################################################################################
temp = pd.DataFrame(X_test, columns= X.columns)
temp['Predictions'] = y_pred


df2['pred'] = y_pred
df4.to_csv('MLPredictions.txt', sep='\t', index=False)


conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=pkpccs01orbit;'
                      'Database=VCC;'
                      'Trusted_Connection=no;')


#df1 = df1.groupby(by=['acctno'], as_index=False)['mnth'].count()

df4.to_csv('MLPredictions.txt', sep='\t', index=False)


