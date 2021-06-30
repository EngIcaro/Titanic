"""
Created 03/06/2021
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
def read_convert_label_enconder(base, column, save_path, save_name):

    encoder = preprocessing.LabelEncoder()
    encoder.classes_ = np.load(save_path+save_name+'.npy', allow_pickle=True)
    encoded = encoder.transform(base[column])

    return encoded

def read_convert_one_hot_encoder(base, name_column_output,encoded,save_path,save_name):
    
    ohe = joblib.load(save_path+save_name+'.joblib')
    onehotlabels = ohe.transform(encoded.reshape(encoded.shape[0], 1)).toarray()
    names_column = ohe.get_feature_names([name_column_output])
    aux = 0
    for i in names_column:
        base[i] = onehotlabels[:,aux]
        aux += 1
    return 
#%%
base = pd.read_csv("input/test.csv")
base_test = base.copy()
#%%
# Criando uma nova coluna (Family) que vai ser o somatório de SibSp com Parch
base_test['FamilySize'] = base_test['SibSp'] + base_test['Parch']
#%%
# Criando um nova coluna (DuplicatedTicket) onde será verdadeiro se existir valores repetidos e falso se não existir
base_test['DuplicatedTicket'] = base_test.duplicated(subset=['Ticket'], keep=False)
#%%
encoded = read_convert_label_enconder(base_test, "Sex", "output/", "labelEncoderSex")
read_convert_one_hot_encoder(base_test, "Sex", encoded, "output/", "oneHotEncoderSex")
#%%
base_test["Age"] = base_test["Age"].fillna(base_test["Age"].median())
baby       = base_test[(base_test["Age"].astype(int) >= 0) & (base_test["Age"].astype(int) <  3 )].index
children   = base_test[(base_test["Age"].astype(int) >= 3) & (base_test["Age"].astype(int) < 13)].index
young      = base_test[(base_test["Age"].astype(int) >= 13) & (base_test["Age"].astype(int) < 21)].index
adult      = base_test[(base_test["Age"].astype(int) >= 21) & (base_test["Age"].astype(int) < 40)].index
middle_old = base_test[(base_test["Age"].astype(int) >= 40) & (base_test["Age"].astype(int) < 60)].index
elderly    = base_test[(base_test["Age"].astype(int) >= 60)].index
base_test.loc[baby,("Age")]       = "baby"
base_test.loc[children,("Age")]   = "children"
base_test.loc[young,("Age")]      = "young"
base_test.loc[adult,("Age")]      = "adult"
base_test.loc[middle_old,("Age")] = "middle_old"
base_test.loc[elderly,("Age")]    = "elderly"  
encoded = read_convert_label_enconder(base_test, "Age", "output/", "labelEncoderAge")
read_convert_one_hot_encoder(base_test, "Age", encoded, "output/", "oneHotEncoderAge")
#%%
base_test["Fare"] = np.log1p(base_test.Fare)
base_test["Fare"] = base_test["Fare"].fillna(base_test["Fare"].median())
#%%
base_test["Embarked"] = read_convert_label_enconder(base_test, "Embarked", "output/", "labelEncoderEmbarked")
#%%
base_test['LastName'] = last= base_test.Name.str.extract('^(.+?),', expand = False)
base_test['LastName'] = read_convert_label_enconder(base_test, "LastName", "output/","labelEncoderLastName")
base_test['Title'] = base_test.Name.str.extract('([A-Za-z]+)\.', expand = False)

least_occuring = ['Rev','Dr','Major', 'Col', 'Capt','Jonkheer','Countess']
base_test.Title = base_test.Title.replace(['Ms', 'Mlle','Mme','Lady'], 'Miss')
base_test.Title = base_test.Title.replace(['Countess','Dona'], 'Mrs')
base_test.Title = base_test.Title.replace(['Don','Sir'], 'Mr')

base_test.Title = base_test.Title.replace(least_occuring,'Rare')
base_test['Title'] = read_convert_label_enconder(base_test, "Title", "output/","labelEncoderTitle")

#%% Carregando o modelo
pkl_filename = "output/pickle_model.pkl"
with open(pkl_filename, 'rb') as file:
    gbm = pickle.load(file)
#%% Fazendo as previsões
base_test.drop(['Cabin', 'PassengerId', 'Sex', 'Age','SibSp','Parch','Ticket','Name'],axis=1,inplace=True)
features_x = ['Pclass','Fare','Embarked','FamilySize','DuplicatedTicket','Sex_0',
              'Sex_1','Age_0','Age_1','Age_2','Age_3','Age_4','Age_5','LastName','Title']
#base_test.drop(['Cabin', 'PassengerId', 'Age','SibSp','Parch','Ticket','Name'],axis=1,inplace=True)
#features_x = ['Pclass','Fare','Embarked','FamilySize','DuplicatedTicket','Sex',
#              'Age_0','Age_1','Age_2','Age_3','Age_4','Age_5','LastName','Title']
#%%
predict_test = gbm.predict(base_test[features_x])
#%% Criando o csv da submissão
submission_pred = pd.DataFrame(columns = ["PassengerId", 'Survived'])
submission_pred["PassengerId"] = base["PassengerId"]
submission_pred["Survived"]  = pd.DataFrame(predict_test, columns=['Survived'])
submission_pred.to_csv('output/pickle_model_sub.csv', index=False)
# %%
