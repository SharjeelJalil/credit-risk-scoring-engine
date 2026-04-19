# -*- coding: utf-8 -*-
"""
Model Comparison and Benchmarking
==================================
Benchmarks 5 classification algorithms for credit risk scoring:
  - Logistic Regression (baseline)
  - MLP Neural Network (25, 5)
  - Random Forest (1000 trees)
  - Gradient Boosting
  - XGBoost

Uses 10-fold cross-validation on F1 score for model selection.
Also contains the full production pipeline: scoring, risk categorization,
proxy income estimation, and limit determination with the risk x salary
band multiplier matrix.

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

#df2['Salary Bracket'] = pd.cut(df2['mth_basic_salary'], [-1, 20000, 25000, 30000, 35000, 40000, 45000, 50000, np.inf], labels=['<=20', '21-25', '26-30', '31-35', '36-40', '41-45', '46-50','>=50'])
#df2.to_csv('CCComplete(SalaryBracket).csv', sep=',', index=False)

########################################################################################
#
#import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
#
#x = df2['Salary Bracket']
#y = df2['Defaulter'].value_count()
#
#sns.axes_style('white')
#sns.set_style('white')
#
#colors = ['pink' if _y >=0 else 'red' for _y in y]
#
#colors = ['pink' if _y >=0 else 'red' for _y in y]
#
#
#ax = sns.barplot(x, y, palette=colors)
#
#for n, (label, _y) in enumerate(zip(x, y)):
#    ax.annotate(
#        s='{:.1f}'.format(abs(_y)),
#        xy=(n, _y),
#        ha='center',va='center',
#        xytext=(0,10),
#        textcoords='offset points',
#        color=color,
#        weight='bold'
#    )
#
#    ax.annotate(
#        s=label,
#        xy=(n, 0),
#        ha='center',va='center',
#        xytext=(0,10),
#        textcoords='offset points',
#    )  
## axes formatting
#ax.set_yticks([])
#ax.set_xticks([])
#sns.despine(ax=ax, bottom=True, left=True)
#


########################################################################################

df3 = pd.read_csv('VCCCustomers2.txt',sep='\t',converters = {'ID_NUMBER':str, 'ACCOUNT_NUM':str})
#df3['EB_CUS_NATIONALITY'].value_counts()


df4 = pd.merge(df2,
               df3,
               left_on='cnic',
               right_on='ID_NUMBER',
               how='inner')

###############################################################################################33

SelectedColumns = ['mth_basic_salary' 'Salary Bracket', 'Defaulter','BankingGroup' , 'ADDRESS_TYPE', 'gender', 'ProductType' , 'relationship' , 'accountOpAge' , 'nowAge' , 'Avg_Bal_Month5' , 'CURRENT_ACCT' , 'SAVING_ACCT' , 'avg_deposit_bal' , 'SMS_FACILITY' , 'NO_OF_LOANS' , 'INACTIVE_CR_CARD' , 'ACTIVE_CR_CARD' , 'INTERNET_BANKING' , 'NO_OF_POLICIES' , 'CR_CARD_CUST_LIMIT' , 'sm_debit' , 'dailyATM_amt' , 'weeklyATM_amt' , 'monthlyATM_amt' , 'weeklyPOS' , 'weeklyUBP_amt' , 'monthlyUBP_amt' , 'weeklyUBP' , 'weeklyCCBLP_amt' , 'monthlyCCBLP' , 'dailyFT_out_amt' , 'weeklyFT_out_amt' , 'monthlyFT_out_amt' , 'weeklyFT_out']
dd = [x for x in df4.columns if x not in SelectedColumns]
df4.drop(dd, axis=1,inplace=True)


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



###########################################################################
#Data Distribution
#count_no_default = len(df5[df5['Defaulter']==0])
#count_no_default
#count_default = len(df5[df5['Defaulter']==1])
#count_default
#pct_of_no_default = count_no_default/(count_no_default + count_default)
#print("percentage of no default is", pct_of_no_default*100)
#pct_of_default = count_default/(count_no_default + count_default)
#print("percentage of default", pct_of_default*100)


#######################################################################################

#Defining Predictor and Target
X = df5.loc[:, df5.columns != 'Defaulter']
y = df5['Defaulter']

######################################################################################

x = X.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df6 = pd.DataFrame(x_scaled, columns=X.columns, index=X.index)


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
X_resampled, y_resampled = kmeans_smote.fit_sample(df6, y)

[print('Class {} has {} instances after oversampling'.format(label, count))
 for label, count in zip(*np.unique(y_resampled, return_counts=True))]

########################################################################################

#Splitting Dataset
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=0)

#######################################################################################

#MLP
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(25, 5), random_state=125)
mlp.fit(X_train, y_train)                        
#PreProb = mlp.predict_proba(X_test)[:,0]

filename = 'finalized_model.sav'
joblib.dump(mlp, filename)

# some time later...

# load the model from disk
loaded_model = joblib.load(filename)
result = loaded_model.score(X_test, y_test)
print(result)

y_pred = mlp.predict(X_test)

print('Accuracy of MLP classifier on test set: {:.2f}'.format(mlp.score(X_test, y_test)))

#Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#Classification Report
print(classification_report(y_test, y_pred, digits = 4))


#########################################################################################

#Logistic Regression
from sklearn.linear_model import LogisticRegression
logit = LogisticRegression()
logit.fit(X_train, y_train)
PreProb = logit.predict_proba(X_test)[:,0]
#y_pred = np.where(PreProb > 0.5, 'No','Yes')
y_pred = logit.predict(X_test)

print('Accuracy of Logit classifier on test set: {:.2f}'.format(logit.score(X_test, y_test)))

#Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#Classification Report
print(classification_report(y_test, y_pred, digits = 4))


#########################################################################################

#Random Forest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=1000, random_state=0,n_jobs=-1)
classifier.fit(X_train, y_train)                        
#PreProb = classifier.predict_proba(X_test)[:,0]
y_pred = classifier.predict(X_test)

print('Accuracy of Random Forest classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))

#Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#Classification Report
print(classification_report(y_test, y_pred, digits = 4))

#########################################################################################

#Gradient Boosting
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
classifier = GradientBoostingClassifier()
classifier.fit(X_train, y_train)                        
#PreProb = classifier.predict_proba(X_test)[:,0]
y_pred = classifier.predict(X_test)

print('Accuracy of Gradient Boosting classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))

#Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#Classification Report
print(classification_report(y_test, y_pred, digits = 4))

#########################################################################################

#Extreme Gradient Boosting
import xgboost as xgb
#from xgboost import export_graphviz
import graphviz

#data_dmatrix = xgb.DMatrix(data=X,label=y)
xg_reg = xgb.XGBClassifier(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 10)
xg_reg.fit(X_train,y_train)
y_pred = xg_reg.predict(X_test)

print('Accuracy of XGBoost classifier on test set: {:.2f}'.format(xg_reg.score(X_test, y_test)))

cm = confusion_matrix(y_test, y_pred)
print(cm)

#Classification Report
print(classification_report(y_test, y_pred, digits = 4))

#plot_tree(xg_reg, rankdir='LR')
plot_tree(xg_reg)
plt.show()


#########################################################################################
##Accuracy
#print('Accuracy of classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))
#
##Confusion Matrix
#cm = confusion_matrix(y_test, y_pred)
#print(cm)
#
##Classification Report
#print(classification_report(y_test, y_pred, digits = 4))

######################################################################################

#ROC
#
#import sklearn.metrics as metrics
#
#fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
#roc_auc = metrics.auc(fpr, tpr)
#
#import matplotlib.pyplot as plt
#plt.title('Receiver Operating Characteristic')
#plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
#plt.legend(loc = 'lower right')
#plt.plot([0, 1], [0, 1],'r--')
#plt.xlim([0, 1])
#plt.ylim([0, 1])
#plt.ylabel('True Positive Rate')
#plt.xlabel('False Positive Rate')
#plt.show()
#plt.savefig('XGBTree')
#
{
    'label': 'Logistic Regression',
    'model': LogisticRegression(random_state=0, C=1, class_weight='balanced'),
},
{
    'label': 'Random Forest',
    'model': RandomForestClassifier(n_estimators=1000, random_state=0,n_jobs=-1),
},  
{
    'label': 'Gradient Boosting',
    'model': GradientBoostingClassifier(),
},
{
    'label': 'Extreme Gradient Boosting',
    'model': xgb.XGBClassifier(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 10),
} 
#######################################################################################

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


###############################################################################
###############################################################################
###############################################################################

#CrossValidation

import pandas as pd, os
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier
#from sklearn.svm import LinearSVC
#from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


log_clf_consumer = LogisticRegression(random_state=0, C=1, class_weight='balanced')
rnd_clf_consumer = RandomForestClassifier(n_estimators=1000, random_state=0,n_jobs=-1)
mlp_clf_consumer = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(25, 5), random_state=125)
gbm_clf_consumer = GradientBoostingClassifier()
xgb_clf_consumer = xgb.XGBClassifier(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 10)
#voting_clf_consumer = VotingClassifier(
#                    estimators=[('lr',log_clf_consumer),('rf',rnd_clf_consumer),('svc',svm_clf_consumer)],
#                    voting='hard'
#                    )

CV=10
cv_df_c = pd.DataFrame(index=range(CV * 3))
entries = []
for clf in (log_clf_consumer, rnd_clf_consumer, mlp_clf_consumer, gbm_clf_consumer, xgb_clf_consumer):
    print(clf)
    model_name = clf.__class__.__name__
    print(model_name)
    accuracies = cross_val_score(clf, X_train, y_train, scoring='f1', cv=CV)
    for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))
cv_df_c = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'f1'])
cv_df_c.groupby('model_name').f1.mean()



##################################################################################################
## production

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

#result_set['Risk Category'] = pd.cut(result_set['DefaultProb'], [-1, 0.25, 0.5, 0.75, np.inf], labels=['Very Low Risk', 'Low Risk', 'High Risk', 'Very High Risk'])
result_set['Risk Category'] = pd.cut(result_set['DefaultProb'], [-1, 0.2, 0.4, 0.6, 0.8, np.inf], labels=['Very Low Risk', 'Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk'])
#result_set.to_csv('ProductionResult.csv', sep=',', index=False)

######################################################################################

#KYC Salary

SalaryBrackets = df1.iloc[:,[3,18,26]]
SalaryBrackets = SalaryBrackets.groupby(by=['cnic'], as_index=False).agg({'ag' : 'max','mth_basic_salary' : 'max'})
#SalaryBrackets['Salary Bracket'] = pd.cut(SalaryBrackets['mth_basic_salary'], [-1, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, np.inf], labels=['<=20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100', '>=100000'])

def Mapping(x):
    if x > 0:
        return 'Yes'
    else :
        return 'No'

SalaryBrackets['Defaulter'] = [Mapping(x) for x in SalaryBrackets["ag"]]
SalaryBrackets.to_csv('CCComplete(SalaryBracket).csv', sep=',', index=False)

SalaryBrackets['cnic'] = [str(x).replace('-','') for x in SalaryBrackets['cnic']]

Salary_Prob_Binned = pd.merge(result_set,
                              SalaryBrackets,
                              left_on='ID_NUMBER',
                              right_on='cnic',
                              how='inner')

#Salary_Prob_Binned.to_csv('SalaryProbBinnedResult.csv', sep=',', index=False)

######################################################################################

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

#
#CCDEPNewSalary = pd.merge(Salary_Prob_Binned,
#                          avgBal_data,
#                          left_on='ACCOUNT_NUM',
#                          right_on='ACCOUNT_NUM',
#                          how='inner')


#CCDEPNewSalary.to_csv('CCDEPNewSalarySalaryProbBinnedResult.csv', sep=',', index=False)


DEPNewSalary = pd.merge(result_set,
                        avgBal_data,
                        left_on='ACCOUNT_NUM',
                        right_on='ACCOUNT_NUM',
                        how='left')


#DEPNewSalary.to_csv('DEPNewSalarySalaryProbBinnedResult.csv', sep=',', index=False)


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
LimitDeterminedFile.to_csv('LimitDeterminedResults.txt', sep='\t', index=False)

        
#    weightage = {'' : ''}
#    
#    for bracket in mainDF['Salary Bracket']:
#        mainDF.loc[(mainDF['Salary Bracket'] == bracket) & mainDF['DefaultProb'] < 0.6, 'Limit'] = mainDF['est_salary'] * 1.5
#        temp = mainDF[mainDF['Salary Bracket'] == bracket].copy()
#        
#    
#    if(SalaryBracket == '21-30').all():
#        if(DProb < 0.6):
#            return EstSalary*1.5
#        elif(DProb >0.6<0.8): 
#            return EstSalary*0.5    
#        else:
#            return EstSalary*0
#
#    elif(SalaryBracket == '31-40').all():
#        if(DProb < 0.6):
#            return EstSalary*1.5
#        elif(DProb >0.6<0.8):    
#            return EstSalary*0.5    
#        else:
#            return EstSalary*0   
#            
#    elif(SalaryBracket == '41-50').all():            
#
#        if(DProb < 0.6):
#            return EstSalary*2
#        elif(DProb >0.6<0.8):    
#            return EstSalary*1    
#        else:
#            return EstSalary*0.5
#            
#            
#    elif(SalaryBracket == '51-60').all():
#        if(DProb < 0.6):
#            return EstSalary*2
#        elif(DProb >0.6<0.8):    
#            return EstSalary*1    
#        else:
#            return EstSalary*0.5            
#            
#            
#    elif(SalaryBracket == '61-70').all():
#        if(DProb < 0.6):
#            return EstSalary*2.5
#        elif(DProb >0.6<0.8):    
#            return EstSalary*1    
#        else:
#            return EstSalary*0.5       
#            
#            
#    elif(SalaryBracket == '71-80').all():
#        if(DProb < 0.6):
#            return EstSalary*3
#        elif(DProb >0.6<0.8):    
#            return EstSalary*1   
#        else:
#            return EstSalary*0.5            
#            
#    elif(SalaryBracket == '81-90').all():
#        if(DProb < 0.6):
#            return EstSalary*4
#        elif(DProb >0.6<0.8):    
#            return EstSalary*1    
#        else:
#            return EstSalary*0.5          
#            
#    elif(SalaryBracket == '91-100').all():
#        if(DProb < 0.6):
#            return EstSalary*4
#        elif(DProb >0.6<0.8):    
#            return EstSalary*1   
#        else:
#            return EstSalary*0.5
#
#    elif(SalaryBracket == '>=100000').all():
#        if(DProb < 0.6):
#            return EstSalary*5
#        elif(DProb >0.6<0.8):    
#            return EstSalary*1    
#        else:
#            return EstSalary*0.5


CCDEPNewSalary['Limit'] = LimitDetermination(CCDEPNewSalary['Salary Bracket'], CCDEPNewSalary['DefaultProb'], CCDEPNewSalary['est_salary'])
            
