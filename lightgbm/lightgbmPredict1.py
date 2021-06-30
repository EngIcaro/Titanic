#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 10:14:56 2020

@author: icaro
"""

#%%
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pickle
import joblib
import lightgbm as lgb
#%%
base = pd.read_csv("input/test.csv")
base_copy = base.copy()
#%%
base.drop('Cabin',axis=1,inplace=True)
base.drop('Name',axis=1,inplace=True)
base.drop('PassengerId',axis=1,inplace=True)
base.drop('Ticket',axis=1,inplace=True)
#%%
encoder = preprocessing.LabelEncoder()
encoder.classes_ = np.load('output/encoder_Sex.npy', allow_pickle=True)
encoded = encoder.transform(base["Sex"])
enc = joblib.load('output/oneHotEncoderSex.joblib')
onehotlabels = enc.transform(encoded.reshape(encoded.shape[0], 1)).toarray()
base["female"] = onehotlabels[:,0]
base["male"] = onehotlabels[:,1]
base.drop('Sex',axis=1,inplace=True)
#%%
base["Age"] = base["Age"].fillna(base["Age"].median())
baby       = base[(base["Age"].astype(int) >= 0) & (base["Age"].astype(int) <  3 )].index
children   = base[(base["Age"].astype(int) >= 3) & (base["Age"].astype(int) < 13)].index
young      = base[(base["Age"].astype(int) >= 13) & (base["Age"].astype(int) < 21)].index
adult      = base[(base["Age"].astype(int) >= 21) & (base["Age"].astype(int) < 40)].index
middle_old = base[(base["Age"].astype(int) >= 40) & (base["Age"].astype(int) < 60)].index
elderly    = base[(base["Age"].astype(int) >= 60)].index
base["Age"][baby] = "baby"
base["Age"][children] = "children"
base["Age"][young] = "young"
base["Age"][adult] = "adult"
base["Age"][middle_old] = "middle_old"
base["Age"][elderly] = "elderly" 
encoder = preprocessing.LabelEncoder()
encoder.classes_ = np.load('output/encoder_Age.npy', allow_pickle=True)
encoded = encoder.transform(base["Age"])
enc = joblib.load('output/oneHotEncoderAge.joblib')
onehotlabels = enc.transform(encoded.reshape(encoded.shape[0], 1)).toarray()
base["adult"]      = onehotlabels[:, 0] 
base["children"]   = onehotlabels[:, 2]
base["young"]      = onehotlabels[:, 5]
base["baby"]       = onehotlabels[:, 1]
base["middle_old"] = onehotlabels[:, 4]
base["elderly"]    = onehotlabels[:, 3]
base.drop('Age',axis=1,inplace=True)
#%%
base["Fare"] = np.log1p(base.Fare)
base["Fare"] = base["Fare"].fillna(base["Fare"].median())
#%%
encoder = preprocessing.LabelEncoder()
encoder.classes_ = np.load('output/encoder_Embarked.npy', allow_pickle=True)
base["Embarked"] = encoder.transform(base["Embarked"])
#%% Carregando o modelo
pkl_filename = "output/lightGBM[2].pkl"
with open(pkl_filename, 'rb') as file:
    gbm = pickle.load(file)
#%% Fazendo as previsões
features_x = ["Pclass", "female","male" , "SibSp", "Parch", "Fare", "Embarked", "adult", "children", "young", "baby", "middle_old", "elderly"]
pred_x = base[features_x]
predict_test = gbm.predict(pred_x)
#%% Criando o csv da submissão
submission_pred = pd.DataFrame(columns = ["PassengerId", 'Survived'])
submission_pred["PassengerId"] = base_copy["PassengerId"]
submission_pred["Survived"]  = pd.DataFrame(predict_test, columns=['Survived'])
submission_pred.to_csv('output/predLightgbm[2].csv', index=False)