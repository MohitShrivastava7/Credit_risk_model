# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
import warnings
import os


# Reading The Data
PROJECT_DIR =r'D:\Ml projects\credil risk modeling\credit risk'
DATA_DIR = 'dataset'

def dataset(name):
    file_name=f'{name}.xlsx'
    file_path = os.path.join(PROJECT_DIR,DATA_DIR,file_name)
    return pd.read_excel(file_path)

df1 = dataset('case_study1')
df1

df2 = dataset('case_study2')
df2

# Remove null
df1 = df1.loc[df1['Age_Oldest_TL']!= -99999]

columns_to_be_removed = []

for i in df2.columns:
    if df2.loc[df2[i]== -99999].shape[0] > 10000:
        columns_to_be_removed.append(i)
        
# Drop columns_to_be_removed columns from df2

df2 = df2.drop(columns_to_be_removed,axis=1)

# Remove null value from df2 

for i in df2.columns:
    df2 = df2.loc[df2[i]!= -99999]
    
# Check for common column

for i in df1.columns:
    if i in df2.columns:
        print(i)
        
# merge these two dataframe

df = df1.merge(df2,on='PROSPECTID')

df.isnull().sum().sum()

# Check no. of categorical columns

for i in df.columns:
    if df[i].dtypes == 'object':
        print(i)
         
# Chi-square test
for i in ['MARITALSTATUS','EDUCATION','GENDER','last_prod_enq2','first_prod_enq2']:
    chi2, pval, _, _ = chi2_contingency(pd.crosstab(df[i], df['Approved_Flag']))
    print(i,'---',pval)
  
# Since all the value is less than significant value 0.05, so we accept all columns

# VIF for numerical col
numeric_col=[]
for i in df.columns:
    if (df[i].dtypes == 'int64') or (df[i].dtypes == 'float64'):
        if i == 'PROSPECTID':
            continue
        numeric_col.append(i)

# VIF Sequentially Check

vif_data = df[numeric_col]
total_col = vif_data.shape[1]
col_to_be_kept = []
column_index=0

for i in range(0,total_col):
    vif_value = variance_inflation_factor(vif_data, column_index)
    print(column_index,'---',vif_value)
    
    if vif_value <=6:
        col_to_be_kept.append(numeric_col[i])
        column_index=column_index + 1 
        
    else:
        vif_data=vif_data.drop([numeric_col[i]],axis=1)
        
# After check now we have only 39 columns

# Check  ANOVA on column_to_be_kept

from scipy.stats import f_oneway
column_to_be_kept_numerical=[]

for i in col_to_be_kept:
    a=list(df[i])
    b=list(df['Approved_Flag'])
    
    group_p1 = [value for value,group in zip(a,b) if group == 'P1']
    group_p2 = [value for value,group in zip(a,b) if group == 'P2']
    group_p3 = [value for value,group in zip(a,b) if group == 'P3']
    group_p4 = [value for value,group in zip(a,b) if group == 'P4']
    
    f_statsistic,p_value = f_oneway(group_p1,group_p2,group_p3,group_p4)
    
    if p_value <=0.05:
        column_to_be_kept_numerical.append(i)
        
        
# Feature Selection is done

# listing all the final features
features = column_to_be_kept_numerical + ['MARITALSTATUS', 'EDUCATION', 'GENDER' , 'last_prod_enq2' ,'first_prod_enq2']
df = df[features + ['Approved_Flag']]

# Label encoding for the categorical features

['MARITALSTATUS', 'EDUCATION', 'GENDER' , 'last_prod_enq2' ,'first_prod_enq2']



df['MARITALSTATUS'].unique()    
df['EDUCATION'].unique()
df['GENDER'].unique()
df['last_prod_enq2'].unique()
df['first_prod_enq2'].unique()



# Ordinal feature -- EDUCATION
# SSC            : 1
# 12TH           : 2
# GRADUATE       : 3
# UNDER GRADUATE : 3
# POST-GRADUATE  : 4
# OTHERS         : 1
# PROFESSIONAL   : 3


# Others has to be verified by the business end user 

df.loc[df['EDUCATION'] == 'SSC',['EDUCATION']] = 1
df.loc[df['EDUCATION'] == '12TH',['EDUCATION']] = 2 
df.loc[df['EDUCATION'] == 'GRADUATE',['EDUCATION']] = 3
df.loc[df['EDUCATION'] == 'UNDER GRADUATE',['EDUCATION']] = 3
df.loc[df['EDUCATION'] == 'POST-GRADUATE',['EDUCATION']] = 4
df.loc[df['EDUCATION'] == 'OTHERS',['EDUCATION']] = 1
df.loc[df['EDUCATION'] == 'PROFESSIONAL',['EDUCATION']] = 3


df['EDUCATION'].value_counts()
df['EDUCATION'] = df['EDUCATION'].astype('int64')
df.info()

df_encoded = pd.get_dummies(df,columns = ['MARITALSTATUS', 'GENDER' , 'last_prod_enq2' ,'first_prod_enq2'])

df_encoded.info()
k = df_encoded.describe()
k

# Data Processing

# 1. Random Forest

y = df_encoded['Approved_Flag']
x = df_encoded.drop(['Approved_Flag'],axis=1)

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=42 )

rfc = RandomForestClassifier(n_estimators=200,random_state=42)
rfc.fit(x_train,y_train)
y_pred = rfc.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print()
print(f'Accuracy : {accuracy:.2f}')
print()
precision,recall,f1_score,_ = precision_recall_fscore_support(y_test, y_pred)


for i, v in enumerate(['p1', 'p2', 'p3', 'p4']):
    print(f"Class {v}:")
    print(f"Precision: {precision[i]}")
    print(f"Recall: {recall[i]}")
    print(f"F1 Score: {f1_score[i]}")
    print()
    
# XGboost

import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

xgbc = xgb.XGBClassifier(objective='multi:softmax', num_class=4)

y = df_encoded['Approved_Flag']
x = df_encoded.drop(['Approved_Flag'],axis=1)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

x_train,x_test,y_train,y_test = train_test_split(x,y_encoded, test_size=0.2,random_state=42 )

xgbc.fit(x_train,y_train)
y_pred = xgbc.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print()
print(f'Accuracy : {accuracy:.2f}')
print()
precision,recall,f1_score,_ = precision_recall_fscore_support(y_test, y_pred)


for i, v in enumerate(['p1', 'p2', 'p3', 'p4']):
    print(f"Class {v}:")
    print(f"Precision: {precision[i]}")
    print(f"Recall: {recall[i]}")
    print(f"F1 Score: {f1_score[i]}")
    print()
    
    
# decision tree

from sklearn.tree import DecisionTreeClassifier

y = df_encoded['Approved_Flag']
x = df_encoded. drop ( ['Approved_Flag'], axis = 1 )

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


dt_model = DecisionTreeClassifier(max_depth=20, min_samples_split=10)
dt_model.fit(x_train, y_train)
y_pred = dt_model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print ()
print(f"Accuracy: {accuracy:.2f}")
print ()

precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)

for i, v in enumerate(['p1', 'p2', 'p3', 'p4']):
    print(f"Class {v}:")
    print(f"Precision: {precision[i]}")
    print(f"Recall: {recall[i]}")
    print(f"F1 Score: {f1_score[i]}")
    print()
    
    
# Hyperparameter tuning in xgboost
# from sklearn.model_selection import GridSearchCV
x_train,x_test,y_train,y_test = train_test_split(x,y_encoded,test_size=.2,random_state=42)

# Define the XGBClassifier with the initial set of hyperparameters

# Define the hyperparameter grid
param_grid = {
   'colsample_bytree': [0.1, 0.3, 0.5, 0.7, 0.9],
  'learning_rate'   : [0.001, 0.01, 0.1, 1],
   'max_depth'       : [3, 5, 8, 10],
   'alpha'           : [1, 10, 100],
  'n_estimators'    : [10,50,100]
}


index = 0

answers_grid = {
     'combination'       :[],
     'train_Accuracy'    :[],
     'test_Accuracy'     :[],
     'colsample_bytree'  :[],
     'learning_rate'     :[],
     'max_depth'         :[],
     'alpha'             :[],
     'n_estimators'      :[]

     }


# Loop through each combination of hyperparameters
for colsample_bytree in param_grid['colsample_bytree']:
  for learning_rate in param_grid['learning_rate']:
    for max_depth in param_grid['max_depth']:
      for alpha in param_grid['alpha']:
          for n_estimators in param_grid['n_estimators']:
             
              index = index + 1
             
              # Define and train the XGBoost model
              model = xgb.XGBClassifier(objective='multi:softmax',  
                                       num_class=4,
                                       colsample_bytree = colsample_bytree,
                                       learning_rate = learning_rate,
                                       max_depth = max_depth,
                                       alpha = alpha,
                                       n_estimators = n_estimators)
              y = df_encoded['Approved_Flag']
              x = df_encoded. drop ( ['Approved_Flag'], axis = 1 )

              label_encoder = LabelEncoder()
              y_encoded = label_encoder.fit_transform(y)


              x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42)


              model.fit(x_train, y_train)
              
              # Predict on training and testing sets
              y_pred_train = model.predict(x_train)
              y_pred_test = model.predict(x_test)
      
      
             # Calculate train and test results
             
              train_accuracy =  accuracy_score (y_train, y_pred_train)
              test_accuracy  =  accuracy_score (y_test , y_pred_test)
              
              
              # Include into the lists
              answers_grid ['combination']   .append(index)
              answers_grid ['train_Accuracy']    .append(train_accuracy)
              answers_grid ['test_Accuracy']     .append(test_accuracy)
              answers_grid ['colsample_bytree']   .append(colsample_bytree)
              answers_grid ['learning_rate']      .append(learning_rate)
              answers_grid ['max_depth']          .append(max_depth)
              answers_grid ['alpha']              .append(alpha)
              answers_grid ['n_estimators']       .append(n_estimators)
              
                   
              # Print results for this combination
              print(f"Combination {index}")
              print(f"colsample_bytree: {colsample_bytree}, learning_rate: {learning_rate}, max_depth: {max_depth}, alpha: {alpha}, n_estimators: {n_estimators}")
              print(f"Train Accuracy: {train_accuracy:.2f}")
              print(f"Test Accuracy : {test_accuracy :.2f}")
              print("-" * 30)
     
             
result = pd.DataFrame(answers_grid) 
result.to_csv('credit_report.csv',index=False) 
          
              
              
                 