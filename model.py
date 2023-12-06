# -*- coding: utf-8 -*-
#Importing the required Libraries
import pandas as pd
import numpy as np

import pickle
from sklearn.linear_model import LinearRegression  
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer


#Load Data
df = pd.read_csv("train.csv")


#Drop the id column
df = df.drop('p_id', axis=1)


#Need to replace all the '0' values with null 
df[['glucose_concentration', 'blood_pressure', 'skin_fold_thickness', 'serum_insulin', 'bmi']] = df[['glucose_concentration', 'blood_pressure', 'skin_fold_thickness', 'serum_insulin', 'bmi']].replace( 0, np.NaN)


#As in BMI there are only some missing values,
# so instead of doing any thing else we will replace the null values by their mean
impute_bmi = SimpleImputer(missing_values=np.NaN, strategy='mean')
impute_bmi = impute_bmi.fit(df[['bmi']])
df[['bmi']] = impute_bmi.transform(df[['bmi']])

impute_glucose = SimpleImputer(missing_values=np.NaN, strategy='mean')
impute_glucose = impute_glucose.fit(df[['glucose_concentration']])
df[['glucose_concentration']] = impute_glucose.transform(df[['glucose_concentration']])
    

data = df[['bmi', 'age', 'blood_pressure']].copy()
data.dropna(inplace=True)
X_bp = data[['bmi', 'age']].values
y_bp = data['blood_pressure'].values
X_train_bp, X_test_bp, y_train_bp, y_test_bp = train_test_split(X_bp, y_bp, test_size=0.25, random_state = 1)
regression = LinearRegression()
regression.fit(X_train_bp,y_train_bp)
def missing_bp(cols):
    bp_y = cols['blood_pressure']
    age = cols['age']
    bmi = cols['bmi']
    dic = {'bmi':[bmi], 'age':[age]}
    df_dic = pd.DataFrame(dic)
    if pd.isnull(bp_y):
        return regression.predict(df_dic)
    else:
        return bp_y
df['blood_pressure'] = df[['age','bmi', 'blood_pressure']].apply(missing_bp, axis=1)
df['blood_pressure'] = df['blood_pressure'].astype(int)



df.loc[(df['diabetes'] == 0 ) & (df['skin_fold_thickness'].isnull()), 'skin_fold_thickness'] = 27
df.loc[(df['diabetes'] == 1 ) & (df['skin_fold_thickness'].isnull()), 'skin_fold_thickness'] = 32

df.loc[(df['diabetes'] == 0 ) & (df['no_times_pregnant']>10), 'no_times_pregnant'] = 2
df.loc[(df['diabetes'] == 1 ) & (df['no_times_pregnant']>10), 'no_times_pregnant'] = 4

df.loc[(df['diabetes'] == 0 ) & (df['blood_pressure']<40), 'blood_pressure'] = 70
df.loc[(df['diabetes'] == 1 ) & (df['blood_pressure']<40), 'blood_pressure'] = 74.5
df.loc[(df['diabetes'] == 0 ) & (df['blood_pressure']>103), 'blood_pressure'] = 70
df.loc[(df['diabetes'] == 1 ) & (df['blood_pressure']>103), 'blood_pressure'] = 74.5

df.loc[(df['diabetes'] == 0 ) & (df['skin_fold_thickness']>40), 'skin_fold_thickness'] = 27
df.loc[(df['diabetes'] == 1 ) & (df['skin_fold_thickness']>40),'skin_fold_thickness'] = 32
df.loc[(df['diabetes'] == 0 ) & (df['skin_fold_thickness']<20), 'skin_fold_thickness'] = 27
df.loc[(df['diabetes'] == 1 ) & (df['skin_fold_thickness']<20), 'skin_fold_thickness'] = 32

df.loc[(df['diabetes'] == 0 ) & (df['bmi']>48), 'bmi'] = 30.1
df.loc[(df['diabetes'] == 1 ) & (df['bmi']>48), 'bmi'] = 34.3

df.loc[(df['diabetes'] == 0 ) & (df['diabetes pedigree']>1), 'diabetes pedigree'] = 0.3265
df.loc[(df['diabetes'] == 1 ) & (df['diabetes pedigree']>1), 'diabetes pedigree'] = 0.4320

df.loc[(df['diabetes'] == 0 ) & (df['age']>61), 'age'] = 27
df.loc[(df['diabetes'] == 1 ) & (df['age']>61), 'age'] = 36


df=df.drop('serum_insulin', axis=1)
X = df.drop(['diabetes'], axis=1)
y = df['diabetes']
classifier=RandomForestClassifier()
classifier.fit(X,y)


pickle.dump(classifier, open('model.pkl', 'wb'))


