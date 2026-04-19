# -*- coding: utf-8 -*-
"""
MLP Neural Network Training and Production Scoring Pipeline
============================================================
Final production model for the Credit Risk Scoring system. Trains a Multi-Layer
Perceptron Neural Network (25, 5 hidden layers, Adam optimizer) on KMeans-SMOTE
balanced data, serializes the trained model and preprocessing artifacts with
joblib, then scores the full 891K customer base.

Includes the complete downstream pipeline:
  - Default probability prediction and 5-tier risk categorization
  - Proxy income estimation from 6-month average deposit balance (SQL query)
  - Credit limit determination using risk x salary band multiplier matrix

@author: sharjeel.jalil
"""



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
from imblearn.datasets import fetch_datasets
from kmeans_smote import KMeansSMOTE
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn import metrics
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import pyodbc as od
from sklearn.externals import joblib



os.chdir('E:\Data')

df1 = pd.read_excel('Cards Portfolio Jan 2019.xlsx', sheetname='Cards Portfolio Jan 2019', converters = {'cnic':str, 'mth_basic_salary':int, 'ACCOUNT_NUM':str})
df2 = df1.iloc[:,[3,18]]
df2['cnic'] = [str(x).replace('-','') for x in df1['cnic']]
df2= df2.groupby(by=['cnic'], as_index=False).agg({'ag' : 'max'})



SalaryBrackets = df1.iloc[:,[3,18,26]]
SalaryBrackets = SalaryBrackets.groupby(by=['cnic'], as_index=False).agg({'ag' : 'max','mth_basic_salary' : 'max'})
SalaryBrackets['Salary Bracket'] = pd.cut(SalaryBrackets['mth_basic_salary'], [-1, 20000, 25000, 30000, 35000, 40000, 45000, 50000, np.inf], labels=['<=20', '21-25', '26-30', '31-35', '36-40', '41-45', '46-50','>=50'])
SalaryBrackets.to_csv('CCComplete(SalaryBracket).csv', sep=',', index=False)

def Mapping(x):
    if x > 0:
        return 'Yes'
    else :
        return 'No'


df2['Defaulter'] = [Mapping(x) for x in df2["ag"]]
df2.drop('ag', axis=1,inplace=True)

########################################################################################

df3 = pd.read_csv('VCCCustomers2.txt',sep='\t',converters = {'ID_NUMBER':str, 'ACCOUNT_NUM':str})

df4 = pd.merge(df2,
               df3,
               left_on='cnic',
               right_on='ID_NUMBER',
               how='inner')

###############################################################################################33

SelectedColumns = ['mth_basic_salary' 'Salary Bracket', 'Defaulter','BankingGroup' , 'ADDRESS_TYPE', 'gender', 'ProductType' , 'relationship' , 'accountOpAge' , 'nowAge' , 'Avg_Bal_Month5' , 'CURRENT_ACCT' , 'SAVING_ACCT' , 'avg_deposit_bal' , 'SMS_FACILITY' , 'NO_OF_LOANS' , 'INACTIVE_CR_CARD' , 'ACTIVE_CR_CARD' , 'INTERNET_BANKING' , 'NO_OF_POLICIES' , 'CR_CARD_CUST_LIMIT' , 'sm_debit' , 'dailyATM_amt' , 'weeklyATM_amt' , 'monthlyATM_amt' , 'weeklyPOS' , 'weeklyUBP_amt' , 'monthlyUBP_amt' , 'weeklyUBP' , 'weeklyCCBLP_amt' , 'monthlyCCBLP' , 'dailyFT_out_amt' , 'weeklyFT_out_amt' , 'monthlyFT_out_amt' , 'weeklyFT_out']
dd = [x for x in df4.columns if x not in SelectedColumns]
df4.drop(dd, axis=1,inplace=True)

#Pipelining
filename = 'SelectedColumns.sav'
joblib.dump(SelectedColumns, filename)


x = pd.get_dummies(df4[['BankingGroup', 'ProductType', 'ADDRESS_TYPE', 'gender']])
df5 = pd.concat([df4, x], sort=False, axis =1)


df5.drop('BankingGroup', axis=1,inplace=True)
df5.drop('ProductType', axis=1,inplace=True)
df5.drop('ADDRESS_TYPE', axis=1,inplace=True)
df5.drop('gender', axis=1,inplace=True)


df5.fillna(0, inplace=True)

from sklearn.preprocessing import LabelEncoder
lb_category = LabelEncoder()
df5['Defaulter'] = lb_category.fit_transform(df5['Defaulter'])

#Pipelining
filename = 'LabelEncoder.sav'
joblib.dump(lb_category, filename)



#######################################################################################

#Defining Predictor and Target
X = df5.loc[:, df5.columns != 'Defaulter']
y = df5['Defaulter']

######################################################################################

#MinMaxScaler

x = X.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df6 = pd.DataFrame(x_scaled, columns=X.columns, index=X.index)

#Pipelining
filename = 'MinMaxScaler.sav'
joblib.dump(min_max_scaler, filename)
#print(result)


######################################################################################

#Kmeans-Smote
kmeans_smote = KMeansSMOTE(
    kmeans_args={
        'n_clusters': 100
    },
    smote_args={
        'k_neighbors': 10
    }
)
X_resampled, y_resampled = kmeans_smote.fit_sample(df6, y)

########################################################################################

#Splitting Dataset
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=0)

#######################################################################################

#MLP
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(25, 5), random_state=125)
mlp.fit(X_train, y_train) 
y_pred = mlp.predict(X_test)
                       
#Pipelining
filename = 'finalized_model.sav'
joblib.dump(mlp, filename)
loaded_model = joblib.load(filename)
result = loaded_model.score(X_test, y_test)
print(result)

#Accuracy
print('Accuracy of MLP classifier on test set: {:.2f}'.format(mlp.score(X_test, y_test)))

#Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#Classification Report
print(classification_report(y_test, y_pred, digits = 4))

##################################################################################################
models = [
{
    'label': 'MLP Neural Network',
    'model': MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(25, 5), random_state=125),
}

]

# Below for loop iterates through your models list
for m in models:
    model = m['model'] # select the model
    model.fit(X_train, y_train) # train the model
    y_pred=model.predict(X_test) # predict the test data
# Compute False postive rate, and True positive rate
    fpr, tpr, thresholds = metrics.roc_curve(y_test, model.predict_proba(X_test)[:,1])
# Calculate Area under the curve to display on the plot
    auc = metrics.roc_auc_score(y_test,model.predict(X_test))
# Now, plot the com0puted values
    plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (m['label'], auc))
# Custom settings for the plot 
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity(False Positive Rate)')
plt.ylabel('Sensitivity(True Positive Rate)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()   # Display




#Production

evalSet = df3.copy()
removeSet = dd.copy()
idNO = evalSet[['ACCOUNT_NUM','ID_NUMBER']]
removeSet.remove('cnic')
evalSet.drop(removeSet, axis=1,inplace=True)

xp = pd.get_dummies(evalSet[['BankingGroup', 'ProductType', 'ADDRESS_TYPE', 'gender']])
evalSet = pd.concat([evalSet, xp], sort=False, axis =1)
evalSet.drop(['BankingGroup', 'ProductType', 'ADDRESS_TYPE', 'gender','ProductType_O'], axis=1, inplace = True)

evalSet.fillna(0, inplace=True)

xp = evalSet.values
x_scaled = min_max_scaler.transform(xp)
evalSet = pd.DataFrame(x_scaled, columns=evalSet.columns, index=evalSet.index)

evalSetProb = mlp.predict_proba(evalSet.values)[:,1]
evalSet_y_pred = mlp.predict(evalSet.values)

result_set = idNO.copy()
result_set['DefaultProb'] = evalSetProb
result_set['PredictedStatus'] = evalSet_y_pred

result_set['Risk Category'] = pd.cut(result_set['DefaultProb'], [-1, 0.2, 0.4, 0.6, 0.8, np.inf], labels=['Very Low Risk', 'Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk'])
#result_set.to_csv('ProductionResult.csv', sep=',', index=False)


######################################################################################

#6MonthAvgBalance

import pyodbc as od
import sqlalchemy
conn = od.connect("DSN=DB Connector",UID = 'amber', PWD = 'BALF54321!')
conn = sqlalchemy.create_engine('mssql+pyodbc://amber:BALF54321!@PKPCCS01ORBIT/Facebook?driver=SQL+SERVER') 

avgQ = '''
SELECT [ACCOUNT_NUM],avg([Avg_Bal]) as est_salary
  FROM [ADV_CC].[dbo].[customer_deposits] cd
  where 1=1
  and [Month_deposit] >= DateAdd(month, -6, Convert(date, datefromparts(YEAR(GETDATE()),MONTH(GETDATE()),1))) 
  and [Month_deposit] < datefromparts(YEAR(GETDATE()),MONTH(GETDATE()),1)
  and (len(id_number) = 13 and patindex('[^0-9]',id_number) = 0)
  group by ACCOUNT_NUM
'''
avgBal_data = pd.read_sql(avgQ, conn)

DEPNewSalary = pd.merge(result_set,
                        avgBal_data,
                        left_on='ACCOUNT_NUM',
                        right_on='ACCOUNT_NUM',
                        how='left')

######################################################################################

#LimitDetermination

def LimitDetermination(mainDF):
    
    mapping = pd.DataFrame({
            'RiskBand' : ['Low-Medium','High','Very High','Low-Medium','High','Very High','Low-Medium','High','Very High','Low-Medium','High','Very High','Low-Medium','High','Very High','Low-Medium','High','Very High','Low-Medium','High','Very High','Low-Medium','High','Very High','Low-Medium','High','Very High'],
            'SalaryBand' : ['>20>30','>20>30','>20>30','>30>40','>30>40','>30>40','>40>50','>40>50','>40>50','>50>60','>50>60','>50>60','>60>70','>60>70','>60>70','>70>80','>70>80','>70>80','>80>90','>80>90','>80>90','>90>100','>90>100','>90>100','>100','>100','>100'],
            'Multiplier' : [1.5,0.5,0,1.5,0.5,0,2,1,0.5,2,1,0.5,2.5,1,0.5,3,1,0.5,4,1,0.5,4,1,0.5,5,1,0.5]})
    
    mainDF['SalaryBand'] = pd.cut(mainDF['est_salary'], [-1, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, np.inf], labels=['<=20', '>20>30', '>30>40', '>40>50', '>50>60', '>60>70', '>70>80', '>80>90', '>90>100', '>100'])
    mainDF['RiskBand'] = pd.cut(mainDF['DefaultProb'], [0, 0.6, 0.8, 1], labels= ['Low-Medium', 'High', 'Very High'])
    
    
    mainDF = pd.merge(mainDF, mapping, how = 'left', on= ['SalaryBand', 'RiskBand'])
    mainDF['Limit'] = mainDF['est_salary'] * mainDF['Multiplier']
            
    return mainDF

LimitDeterminedFile = LimitDetermination(DEPNewSalary.copy())
#LimitDeterminedFile.to_csv('LimitDeterminedResults.csv', sep=',', index=False)
            
