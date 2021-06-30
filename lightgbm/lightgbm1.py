#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 18:33:09 2020

@author: icaro
"""
#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pickle
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
#%% Lendo a base de teste
base = pd.read_csv("output/train_clean[2]")
#%%
features_x = ["Pclass", "female","male" , "SibSp", "Parch", "Fare", "Embarked", "adult", "children", "young", "baby", "middle_old", "elderly"]
target_y = ["Survived"]
#%%
# Number of boosted trees to fit
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 10)]
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
# Maximum tree leaves for base learners
num_leaves = [int(x) for x in np.linspace(10, 150, num = 15)]
# Boosting learning rate
learning_rate = [0.03, 0.05, 0.1, 0.2, 0.3]
# Number of samples for constructing bins.
subsample_for_bin = [100000,200000, 300000]
# LEMBRAR DE MUDAR O VERIFICAR 
objective = ['binary']
random_grid = {'n_estimators': n_estimators,
               'max_depth': max_depth,
               'num_leaves': num_leaves,
               'learning_rate': learning_rate,
               'subsample_for_bin': subsample_for_bin}
#%%
gbm = lgb.LGBMClassifier()
rf_random = RandomizedSearchCV(estimator = gbm, param_distributions = random_grid, n_iter = 350, cv = 3, verbose = 2, random_state = 42, n_jobs = -1)
rf_random.fit(base[features_x], base[target_y])
#%%
print(rf_random.best_params_)

#%% Agora que achamos os melhores hyperparâmetros de forma aleatória, vamos refinar mais e buscar por hyperparâmetros mais próximos  
param_grid = { 
    'subsample_for_bin': [100000], 
    'num_leaves': [100, 110, 120, 130], 
    'n_estimators': [211,311,411], 
    'max_depth': [8,10,12], 
    'learning_rate': [0.02,0.03, 0.04] 
}
# Crie um modelo baseado 
gbm = lgb.LGBMClassifier()
# Instancie o modelo de pesquisa de grade 
grid_search = GridSearchCV(estimator = gbm, param_grid = param_grid, 
                          cv = 3,n_jobs = -1, verbose = 2)
grid_search.fit(base[features_x], base[target_y])
#%%
print(grid_search.best_params_)
#%% Treinando o modelo com toda a base de dado
all_train_x = base[features_x]
all_train_y = base[target_y]
gbm = lgb.LGBMClassifier(learning_rate= 0.04, max_depth= 10, n_estimators= 211, num_leaves= 100, subsample_for_bin= 100000)
gbm.fit(all_train_x, all_train_y.values.ravel())

pkl_filename = "output/lightGBM[2].pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(gbm, file)