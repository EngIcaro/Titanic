"""
Created: 01/06/2021
@author: icaro
"""
#%%
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
#%%
# Leitura da base de treino
base = pd.read_csv("input/train.csv")
#%%
# Obtendo informações iniciais do projeto
base.info()
#%% Contando Survived and Non Survived
# Survived == 1:  342 Survived == 0:  549
sns.countplot(base.Survived)
print("Survived == 1: ", base[base["Survived"] == 1]["Survived"].count())
print("Survived == 0: ", base[base["Survived"] == 0]["Survived"].count())
#%% Contando Pclass 
# Pclass == 1: 216 | Pclass == 2:  184 | Pclass == 3:  491
sns.countplot(base.Pclass)
print("Number of distinct numbers: ", base["Pclass"].unique())
print("Pclass == 1: ", base[base["Pclass"] == 1]["Pclass"].count())
print("Pclass == 2: ", base[base["Pclass"] == 2]["Pclass"].count())
print("Pclass == 3: ", base[base["Pclass"] == 3]["Pclass"].count())
#%% Quanto maior a classe econômica mais chance tem de sobreviver ? 
sns.countplot(base.Pclass, hue=base.Survived)
print("For Pclass = 1 we have: ")
print(((base[(base["Pclass"] == 1) & (base["Survived"] == 1)]["Survived"].count())/(base[base["Pclass"] == 1]["Survived"].count())),"Survived")
print((1-(base[(base["Pclass"] == 1) & (base["Survived"] == 1)]["Survived"].count())/(base[base["Pclass"] == 1]["Survived"].count()))," Not Survived")
print("For Pclass = 2 we have: ")
print(((base[(base["Pclass"] == 2) & (base["Survived"] == 1)]["Survived"].count())/(base[base["Pclass"] == 2]["Survived"].count())),"Survived")
print((1-(base[(base["Pclass"] == 2) & (base["Survived"] == 1)]["Survived"].count())/(base[base["Pclass"] == 2]["Survived"].count()))," Not Survived")
print("For Pclass = 3 we have: ")
print(((base[(base["Pclass"] == 3) & (base["Survived"] == 1)]["Survived"].count())/(base[base["Pclass"] == 3]["Survived"].count())),"Survived")
print((1-(base[(base["Pclass"] == 3) & (base["Survived"] == 1)]["Survived"].count())/(base[base["Pclass"] == 3]["Survived"].count()))," Not Survived")
#%% Contando quantas pessoas de cada sexo e se a relação com o target
plt.figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')
plt.subplot(1, 2, 1)
sns.countplot(base.Sex)
plt.subplot(1, 2, 2)
sns.countplot(base.Sex, hue=base.Survived)
#%% Qual a faixa etária da idade? mais novos ou mais velhos tem mais chances de sobreviver?
# [bebe 2, criança 2 a 11, adolescente, 12 a 19, adulto, 20 a 39, meia idade 40 a 60, terceira idade 60 ]
# Me parece que a idade não influencia muito na sobrevivência.
base_copy = base.copy()
base_copy["Age_name"] = base["Age"]
base_copy  = base_copy[base_copy["Age_name"].notna()]
baby       = base_copy[(base_copy["Age_name"].astype(int) >= 0) & (base_copy["Age_name"].astype(int) <  3 )].index
children   = base_copy[(base_copy["Age_name"].astype(int) >= 3) & (base_copy["Age_name"].astype(int) < 13)].index
young      = base_copy[(base_copy["Age_name"].astype(int) >= 13) & (base_copy["Age_name"].astype(int) < 21)].index
adult      = base_copy[(base_copy["Age_name"].astype(int) >= 21) & (base_copy["Age_name"].astype(int) < 40)].index
middle_old = base_copy[(base_copy["Age_name"].astype(int) >= 40) & (base_copy["Age_name"].astype(int) < 60)].index
elderly    = base_copy[(base_copy["Age_name"].astype(int) >= 60)].index 
base_copy["Age_name"][baby] = "baby"
base_copy["Age_name"][children] = "children"
base_copy["Age_name"][young] = "young"
base_copy["Age_name"][adult] = "adult"
base_copy["Age_name"][middle_old] = "middle_old"
base_copy["Age_name"][elderly] = "elderly"
plt.figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')
plt.subplot(1, 2, 1)
sns.countplot(base_copy.Age_name)
plt.subplot(1, 2, 2)
sns.countplot(base_copy.Age_name, hue=base_copy.Survived)
#%% analisando a variável SibSp
plt.figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')
plt.subplot(1, 2, 1)
sns.countplot(base.SibSp)
plt.subplot(1, 2, 2)
sns.countplot(base.SibSp, hue=base.Survived)
#%% Analisando a variável Parch
plt.figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')
plt.subplot(1, 2, 1)
sns.countplot(base.Parch)
plt.subplot(1, 2, 2)
sns.countplot(base.Parch, hue=base.Survived)
#%% Existe ticket repetido? a tarifa cobrado é o mesmo para cada tipo de ticket
base.info()
print(base.Ticket.nunique())
duplicated = base[base.Ticket.duplicated(keep=False)]
duplicated = duplicated.sort_values(['Ticket'])
#%% 
# Cauda muito longa a direita é necessário deixar a distribuição mais normal(aplicar log1)
plt.figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')
plt.subplot(1, 2, 1)
sns.distplot(base.Fare)
plt.subplot(1, 2, 2)
sns.violinplot(data=base.Fare)
#%% Pessoas que tinham  cabine teve uma taxa de sobrevivência maior que os que n tem
base['HasCabin'] = base.Cabin.notna()
plt.figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')
plt.subplot(1, 2, 1)
sns.countplot(base.HasCabin)
plt.subplot(1, 2, 2)
sns.countplot(base.HasCabin, hue=base.Survived)
#%% Portão de embarque
print(base.Embarked.nunique())
plt.subplot(1, 2, 1)
sns.countplot(base.Embarked)
plt.subplot(1, 2, 2)
sns.countplot(base.Embarked, hue=base.Survived)
