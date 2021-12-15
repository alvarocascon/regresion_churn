#modelo de regresion logistica para predecir churn en una telco
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Limpieza del DF
#carga
df = pd.read_csv('telecom_churn.txt')
#cambiar nombre
df.rename(columns={"Churn_num": "Churn"}, inplace=True)
#se transforma a numerica; 1,0 en vez de yes, no
df['Churn'] = df['Churn'].astype('int64')
#dumificamos y creamos un df con las variables voice y ip (voice mail plan/international calls.
vmp = pd.get_dummies(df['Voice mail plan'],drop_first=True,prefix="voice")
ip = pd.get_dummies(df['International plan'],drop_first=True,prefix="ip")
#eliminamos las columnas que había antes
df.drop(['Voice mail plan','International plan'],axis=1,inplace=True)
#unimos al df los dfs que hemos creado con las variables categóricas
df = pd.concat([df,vmp,ip],axis=1)
df.head()
df.drop('State',axis=1,inplace=True)
print(df.head())

#MATRIZ DE ENTRENAMIENTO
import random
random.seed(113)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('Churn',axis=1),
                                                    df['Churn'], test_size=0.25,
                                                    random_state=101)
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
#predecimos en X_test para ver la confianza del modelo.
predictions = logmodel.predict(X_test)
#accuraccy
dif = abs(predictions - y_test)
sum_dif = sum(dif)
error_rate = sum_dif / len(dif)
accuracy = 1 - error_rate
print(f"accuracy:{accuracy}")
#tabla con:precision, recall,f-1 score, support
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
#probabilidad del modelo para acertar
from sklearn.metrics import roc_auc_score
roc_auc = roc_auc_score(y_test, logmodel.predict_proba(X_test)[:,1])
print(f"Probabilidad{roc_auc}")