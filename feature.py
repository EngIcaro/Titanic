"""
Created: 01/06/2021
@author: icaro
"""
#%%
# Importando as bibliotecas
import pandas as pd
import joblib
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
#%%
# Definindo as funções
def create_label_enconder(base, column, save_path, save_name):

    encoder = preprocessing.LabelEncoder()
    encoded = encoder.fit_transform(base[column])
    np.save(save_path+save_name+'.npy', encoder.classes_)
    encoded = encoded.reshape(encoded.shape[0],1)

    return encoded

def create_one_hot_encoder(base, name_column_output,encoded,save_path,save_name):
    ohe = preprocessing.OneHotEncoder(handle_unknown='ignore')
    ohe.fit(encoded)
    joblib.dump(ohe, save_path+save_name+'.joblib')
    names_column = ohe.get_feature_names([name_column_output])
    onehotlabels = ohe.transform(encoded).toarray()
    aux = 0
    for i in names_column:
        base[i] = onehotlabels[:,aux]
        aux += 1
    return 
def read_convert_label_enconder(base, column, save_path, save_name):

    encoder = preprocessing.LabelEncoder()
    encoder.classes_ = np.load(save_path+save_name+'.npy', allow_pickle=True)
    encoded = encoder.transform(base[column])

    return encoded

def map_lastName_label_encoder():
    base_train = pd.read_csv("input/train.csv")
    base_test = pd.read_csv("input/test.csv")
    base_all = pd.concat([base_train, base_test], axis=0, sort=False)
    base_all['LastName'] = last= base_all.Name.str.extract('^(.+?),', expand = False)
    base_all['LastName']=create_label_enconder(base_all, "LastName", "output/","labelEncoderLastName")
    return 

map_lastName_label_encoder()
#%%
# Fazendo a Leitura da base de dados
base = pd.read_csv("input/train.csv")
#%%
# Fazendoo a cópia da base de dados
base_feature = base.copy()
#%%
# Criando uma nova coluna (Family) que vai ser o somatório de SibSp com Parch
base_feature['FamilySize'] = base_feature['SibSp'] + base_feature['Parch']+1
#%%
# Criando um nova coluna (DuplicatedTicket) onde será verdadeiro se existir valores repetidos e falso se não existir
base_feature['DuplicatedTicket'] = base_feature.duplicated(subset=['Ticket'], keep=False)
#%% Sex
# Como sex tem apenas dois valores categóricos nominais será utilizado label encoding e depois One-Hot
encoded = create_label_enconder(base_feature, "Sex", "output/", "labelEncoderSex")
create_one_hot_encoder(base_feature, "Sex", encoded, "output/", "oneHotEncoderSex")
#%% convertando as idades em seis categorias diferentes. e depois aplciando o label enconding
base_feature["Age"] = base_feature["Age"].fillna(base_feature["Age"].median())
baby       = base_feature[(base_feature["Age"].astype(int) >= 0) &  (base_feature["Age"].astype(int) <  3 )].index
children   = base_feature[(base_feature["Age"].astype(int) >= 3) &  (base_feature["Age"].astype(int) < 13)].index
young      = base_feature[(base_feature["Age"].astype(int) >= 13) & (base_feature["Age"].astype(int) < 21)].index
adult      = base_feature[(base_feature["Age"].astype(int) >= 21) & (base_feature["Age"].astype(int) < 40)].index
middle_old = base_feature[(base_feature["Age"].astype(int) >= 40) & (base_feature["Age"].astype(int) < 60)].index
elderly    = base_feature[(base_feature["Age"].astype(int) >= 60)].index
base_feature.loc[baby,("Age")]       = "baby"
base_feature.loc[children,("Age")]   = "children"
base_feature.loc[young,("Age")]      = "young"
base_feature.loc[adult,("Age")]      = "adult"
base_feature.loc[middle_old,("Age")] = "middle_old"
base_feature.loc[elderly,("Age")]    = "elderly" 
#%%
# Aplicando One Hot Enoncer da variável AGE
encoded = create_label_enconder(base_feature, "Age", "output/", "labelEncoderAge")
create_one_hot_encoder(base_feature, "Age", encoded, "output/", "oneHotEncoderAge")
#%% A distribuição da tarifa cobrado tem uma calda muito pesada para direita. Vou aplciar o log para normalizar
base_feature["Fare"] = np.log1p(base_feature["Fare"])
#%%
# Aplicando label encoder na variável Embarked
base_feature["Embarked"] = base_feature["Embarked"].fillna('S')
base_feature["Embarked"] = create_label_enconder(base_feature, "Embarked", "output/", "labelEncoderEmbarked")
#%%
# Capturando apenas o lastName
base_feature['LastName'] = last= base_feature.Name.str.extract('^(.+?),', expand = False)
base_feature['LastName']= read_convert_label_enconder(base_feature, "LastName", "output/","labelEncoderLastName")
#%%
# Definindo o titulo do passageiro de acordo com o nome
base_feature['Title'] = base_feature.Name.str.extract('([A-Za-z]+)\.', expand = False)

least_occuring = ['Rev','Dr','Major', 'Col', 'Capt','Jonkheer','Countess']
base_feature.Title = base_feature.Title.replace(['Ms', 'Mlle','Mme','Lady'], 'Miss')
base_feature.Title = base_feature.Title.replace(['Countess','Dona'], 'Mrs')
base_feature.Title = base_feature.Title.replace(['Don','Sir'], 'Mr')

base_feature.Title = base_feature.Title.replace(least_occuring,'Rare')
base_feature['Title'] = create_label_enconder(base_feature, "Title", "output/","labelEncoderTitle")
#%%
# excluindo as colunas que não vão ser utilizadas
base_feature.drop(['Cabin', 'PassengerId', 'Sex', 'Age','SibSp','Parch','Ticket','Name'],axis=1,inplace=True)
#base_feature.drop(['Cabin', 'PassengerId', 'Age','SibSp','Parch','Ticket','Name'],axis=1,inplace=True)

#%% Salvando o novo Dataframe
base_feature.to_csv('output/train_clean',index=False)

# %%
