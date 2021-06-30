# titanic

## Descrição
Repositório criado para resolver o desafio Kaggle Titanic. A solução criada envolve quatro scripts, o primeiro (eda.py) realiza toda a parte da análise exploratória dos dados, o segundo (feature.py) faz a limpeza dos dados e utiliza algumas técnicas de feature engineering para criação de novas features, o terceiro script é o model.py, onde é feita a criação do modelo (Random forest, LightGBM), e, por último(predict.py) responsável pela predição da base de teste.

## Instalação
1. Baixe o arquivo zip desse repositório 
2. Instale [virtualenv](https://virtualenv.pypa.io/en/latest/installation.html)
3. Navegue até o repositório onde o arquivo foi extraído e crie um ambiente virtual com `virtual env`
4. Ative o ambiente com `source env/bin/activate`
5. Instale a dependências com `pip install -r requirements.txt`
6. Execute os scripts na seguinte ordem eda.py -> feature.py -> model.py -> predict.py
7. Divirta-se

## Técnicas e modelos utilizados
* Label encoder
* One-Hot encoder
* Feature engineering
* Random Grid
* Grid Seach
* K-fold
* Random forest
* LightGBM
