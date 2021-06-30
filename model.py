"""
Created 03/06/2021
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
base = pd.read_csv("output/train_clean")
#%%
features_x = ['Pclass','Fare','Embarked','FamilySize','DuplicatedTicket','Sex_0',
              'Sex_1','Age_0','Age_1','Age_2','Age_3','Age_4','Age_5','LastName','Title']
#features_x = ['Pclass','Fare','Embarked','FamilySize','DuplicatedTicket','Sex',
#              'Age_0','Age_1','Age_2','Age_3','Age_4','Age_5','LastName','Title']

target_y = ["Survived"]
#%%
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)
#%%
rf = RandomForestClassifier() 
# Pesquisa aleatória de parâmetros, usando validação cruzada de 3 dobras, 
# pesquise em 100 combinações diferentes e use todos os núcleos disponíveis 
rf_random = RandomizedSearchCV (estimator = rf, param_distributions = random_grid, n_iter = 50, cv = 3, verbose = 2, random_state = 42, n_jobs = -1)
rf_random.fit(base[features_x], base[target_y])
print(rf_random.best_params_)
#%%
for i in range(0,3):
    print(rf_random.cv_results_["split{0}_test_score".format(i)])
#{'n_estimators': 1000, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_features': 'auto', 'max_depth': 50, 'bootstrap': False}
#%% Agora que achamos os melhores hyperparâmetros de forma aleatória, vamos refinar mais e buscar por hyperparâmetros mais próximos  
param_grid = { 
    'bootstrap': [True], 
    'max_depth': [60, 80, 100, 120, 140], 
    'max_features': ['sqrt'], 
    'min_samples_leaf' : [2, 4, 6], 
    'min_samples_split': [8, 10, 12], 
    'n_estimators': [600, 800, 1000, 1400] 
}
# Crie um modelo baseado 
rf = RandomForestClassifier()
# Instancie o modelo de pesquisa de grade 
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)
grid_search.fit(base[features_x], base[target_y])
print(grid_search.best_params_)
#%% Agora que achamos os melhores parâmetros vamos avaliar o erro treinando o modelo com divisão de treino e teste simples. feita anteriormente
# Dividindo o conjunto de treino em treino de validação. colocando a semente do random =3 e 
# deixando os conjuntos de treino e teste com a mesma proporção de sobreviventes e não sobreviventes
train, test = train_test_split(base, test_size = 0.2, random_state=3,stratify=base["Survived"])
#print("Survived == 1: ", train[train["Survived"] == 1]["Survived"].count())
#print("Survived == 0: ", train[train["Survived"] == 0]["Survived"].count())
train_x, train_y = train[features_x], train[target_y]
test_x, test_y   = test[features_x], test[target_y]
mdl = RandomForestClassifier(n_estimators=1000, n_jobs=-1, random_state=0)
mdl.fit(train_x, train_y.values.ravel())
p = mdl.predict(test_x)
print('The accuracy of the Logistic Regression is',metrics.accuracy_score(p,test_y))
#%% Agora com os parâmetros encontrados
mdl = RandomForestClassifier(n_estimators=1000, min_samples_split=10, min_samples_leaf=5, max_features='auto', max_depth=10, bootstrap= False)
mdl.fit(train_x, train_y.values.ravel())
p = mdl.predict(test_x)
print('The accuracy of the Logistic Regression is',metrics.accuracy_score(p,test_y))
#%% Treinando o modelo com toda a base de dado
all_train_x = base[features_x]
all_train_y = base[target_y]
mdl = RandomForestClassifier(n_estimators=1400, min_samples_split=8, min_samples_leaf=6, max_features='sqrt', max_depth=60, bootstrap= True)
mdl.fit(all_train_x, all_train_y.values.ravel())

pkl_filename = "output/pickle_model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(mdl, file)
