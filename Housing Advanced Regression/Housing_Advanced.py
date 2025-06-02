# Primero importamos las bibliotecas necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

current_dir = os.path.dirname(os.path.abspath(__file__))

#Cargamos el train y test 
test_df = pd.read_csv(os.path.join(current_dir, 'test.csv'))
train_df = pd.read_csv(os.path.join(current_dir, 'train.csv'))

#procedemos a preprocesar los datos
#primero, separamos las columnas numericas y categoricas
#removiendo la columna SalePrice del train
train_df_numericas = train_df.select_dtypes(include=['int64', 'float64']).drop(columns=['SalePrice'])
train_df_categoricas = train_df.select_dtypes(include=['object'])

#ahora, se realiza un pipeline para procesar las columnas numericas y categoricas

pipeline_numericas = Pipeline([('imputer', SimpleImputer(strategy='mean')),
                              ('scaler', StandardScaler())])

pipeline_categoricas = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                                 ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))])

#tratamos las columnas numericas y categoricas del train
pipeline_numericas.fit_transform(train_df_numericas)
pipeline_categoricas.fit_transform(train_df_categoricas)

