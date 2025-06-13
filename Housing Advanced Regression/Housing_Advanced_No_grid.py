import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
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

#hacemos feature engineering antes de aplicar el pipeline
train_df_numericas['TotalSF'] = train_df_numericas['TotalBsmtSF'] + train_df_numericas['1stFlrSF'] + train_df_numericas['2ndFlrSF']
train_df_numericas['TotalBath'] = train_df_numericas['FullBath'] + 0.5 * train_df_numericas['HalfBath']
train_df_numericas['TotalPorch'] = train_df_numericas['OpenPorchSF'] + train_df_numericas['EnclosedPorch'] + train_df_numericas['3SsnPorch'] + train_df_numericas['ScreenPorch']
train_df_numericas['Age'] = train_df_numericas['YrSold'] - train_df_numericas['YearBuilt']

#ahora, se realiza un pipeline para procesar las columnas numericas y categoricas

pipeline_numericas = Pipeline([('imputer', SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)),
                              ('scaler', StandardScaler())])

pipeline_categoricas = Pipeline([('imputer', SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=None)),
                                 ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))])

#tratamos las columnas numericas y categoricas del train
X_numericas = pipeline_numericas.fit_transform(train_df_numericas)
X_categoricas = pipeline_categoricas.fit_transform(train_df_categoricas)

#Creamos nuestra variable objetivo
Y_train = train_df['SalePrice']

#Ahora, combinamos las columnas numericas y categoricas del train
X_Combined = np.concatenate([X_numericas, X_categoricas], axis=1)

#Procedemos a separar el train en train y validation
X_train, X_val, Y_train, Y_val = train_test_split(X_Combined, Y_train, test_size=0.2, random_state=42)

#Despues de realizar una busqueda de hiperparametros con GridSearchCV, hemos encontrado los mejores parametros
# para el modelo XGBRegressor. Ahora, podemos proceder a entrenar el modelo
# y evaluar su rendimiento en el conjunto de entrenamiento y validación.

parametros = {'colsample_bytree': 0.5, 'learning_rate': 0.05, 'max_depth': 3, 'min_child_weight': 1, 'n_estimators': 900, 'subsample': 0.8}

#Ahora, procedemos a entrenar el modelo y evaluarlo
#sin embargo, antes, aplicamos un GridSearchCV para encontrar los mejores parametros (primera vez)
model_regressor = XGBRegressor(**parametros, random_state=42)

#Entrenamos el modelo
model_regressor.fit(X_train, Y_train)

#Evaluamos el modelo en el conjunto de entrenamiento
train_predictions = model_regressor.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(Y_train, train_predictions))
print(f"RMSE en el conjunto de entrenamiento: {train_rmse}")

#Evaluamos el modelo en el conjunto de validacion
val_predictions = model_regressor.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(Y_val, val_predictions))
print(f"RMSE en el conjunto de validacion: {val_rmse}")


#Como el modelo ha sido entrenado, podemos proceder a realizar predicciones en el conjunto de test
#primero separamos las columnas numericas y categoricas del test
test_df_numericas = test_df.select_dtypes(include=['int64', 'float64'])
#hacemos feature engineering en el test
test_df_numericas['TotalSF'] = test_df_numericas['TotalBsmtSF'] + test_df_numericas['1stFlrSF'] + test_df_numericas['2ndFlrSF']
test_df_numericas['TotalBath'] = test_df_numericas['FullBath'] + 0.5 * test_df_numericas['HalfBath']
test_df_numericas['TotalPorch'] = test_df_numericas['OpenPorchSF'] + test_df_numericas['EnclosedPorch'] + test_df_numericas['3SsnPorch'] + test_df_numericas['ScreenPorch']
test_df_numericas['Age'] = test_df_numericas['YrSold'] - test_df_numericas['YearBuilt']

test_df_categoricas = test_df.select_dtypes(include=['object'])

X_numericas_test = pipeline_numericas.transform(test_df_numericas)
X_categoricas_test = pipeline_categoricas.transform(test_df_categoricas)
X_Combined_test = np.concatenate([X_numericas_test, X_categoricas_test], axis=1)

#predecimos el conjunto de test
test_predictions = model_regressor.predict(X_Combined_test)


# PUNTAJE:  0.13168 V2

