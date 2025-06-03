# Primero importamos las bibliotecas necesarias
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
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

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

#Ahora, procedemos a entrenar el modelo y evaluarlo
#sin embargo, antes, aplicamos un GridSearchCV para encontrar los mejores parametros (primera vez)
model_regressor = XGBRegressor(random_state=42)

#Definimos los parametros a buscar
parametros = {
    'n_estimators': [100, 500,900],
    'learning_rate': [0.01, 0.05, 0.08],
    'max_depth': [3, 7,10],
    'min_child_weight': [1, 2, 5],
    'subsample': [0.5, 0.8, 0.9],
    'colsample_bytree': [0.5, 0.8, 0.9]
}

#Aplicamos GridSearchCV
grid_search = GridSearchCV(
    estimator=model_regressor, 
    param_grid=parametros, 
    cv=5, 
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=2)

grid_search.fit(X_train, Y_train)

#Obtenemos los mejores parametros
mejores_parametros = grid_search.best_params_
print(f"Mejores parametros: {mejores_parametros}")
#Veamos el mejor score de acuerdo a RMSE
print(f"Mejor score de acuerdo a RMSE: {-grid_search.best_score_}")

#Ahora, seleccionamos el mejor modelo
model_regressor = XGBRegressor(**mejores_parametros, random_state=42)

#Evaluamos el modelo en el conjunto de entrenamiento
train_predictions = model_regressor.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(Y_train, train_predictions))
print(f"RMSE en el conjunto de entrenamiento: {train_rmse}")

#Evaluamos el modelo en el conjunto de validacion
val_predictions = model_regressor.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(Y_val, val_predictions))
print(f"RMSE en el conjunto de validacion: {val_rmse}")





