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

current_dir = os.path.dirname(os.path.abspath(__file__))

#Cargamos el train y test 
test_df = pd.read_csv(os.path.join(current_dir, 'test.csv'))
train_df = pd.read_csv(os.path.join(current_dir, 'train.csv'))

#Como buena practica, separamos los datos en numericos y categoricos 
#Realizamos el preprocesamiento de los datos
#tanto para el train como para el test

#Quitamos las columnas que no suman "mucha" informacion para el modelo
drop_cols = ['Id', 'SalePrice','MiscVal','LowQualFinSF','BsmtHalfBath', 'KitchenAbvGr','3SsnPorch','ScreenPorch','PoolArea','MoSold','YrSold','MSSubClass']
cols_numericas = train_df.select_dtypes(include=['int64', 'float64']).columns.drop(drop_cols)
cols_categoricas = train_df.select_dtypes(include=['object']).columns

#Tratamos las columnas numericas y categoricas del train
scaler = StandardScaler()
imputer = SimpleImputer(strategy='mean')
X_numerical = train_df[cols_numericas]
X_numerical = imputer.fit_transform(X_numerical)
X_numerical = scaler.fit_transform(X_numerical)

#Tratamos las columnas categoricas del train
categorical_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_categorical = train_df[cols_categoricas]
X_categorical_encoded = categorical_encoder.fit_transform(X_categorical)

#Combinamos las columnas numericas y categoricas del train
X_Combined = np.concatenate([X_numerical, X_categorical_encoded], axis=1)

#sacamos la columna SalePrice del train
Y_train = train_df['SalePrice']

#para la evaluacion final, usaremos el conjunto de entrenamiento completo
#por lo cual usamos la siguiente linea para conservar el conjunto Y original
Y_train_val_final = Y_train.copy()

#Separamos los datos en train y validation para evitar el sobreajuste
X_train, X_val, Y_train, Y_val = train_test_split(X_Combined, Y_train, test_size=0.2, random_state=1)

#Entrenamos el modelo
model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=4, random_state=1, n_jobs=-1)
model.fit(X_train, Y_train)

#Evaluamos el modelo en el conjunto de entrenamiento
train_predictions = model.predict(X_train)
train_mse = mean_squared_error(Y_train, train_predictions)
print("\nError en datos de entrenamiento (RMSE):", np.sqrt(train_mse))

#Evaluamos el modelo en el conjunto de validation
val_predictions = model.predict(X_val)
val_mse = mean_squared_error(Y_val, val_predictions)
print("Error en datos de validación (RMSE):", np.sqrt(val_mse))

#Predecimos el conjunto de test
#Sin embargo, primero preprocesamos el conjunto de test
test_categorical = test_df[cols_categoricas]
test_numerical = test_df[cols_numericas]

test_numerical = imputer.transform(test_numerical)
test_numerical = scaler.transform(test_numerical)

test_categorical_encoded = categorical_encoder.transform(test_categorical)
test_df = np.concatenate([test_numerical, test_categorical_encoded], axis=1)

#Predecimos el conjunto de test
predictions = model.predict(test_df)

#una vez entrenado el modelo y las predicciones hechas, podemos evaluar el modelo
#usamos el error cuadratico medio para evaluar el modelo
final_predictions = model.predict(X_Combined)  
mse = mean_squared_error(Y_train_val_final, final_predictions)

#imprimimos el error cuadratico medio
print("\nRoot Mean Squared Error:", np.sqrt(mse))

# Realizar validación cruzada con 5 folds
cv_scores = cross_val_score(model, X_Combined, Y_train_val_final, 
                          cv=5, 
                          scoring='neg_root_mean_squared_error',
                          n_jobs=-1)

# Los scores vienen en negativo, así que los convertimos a positivo
cv_scores = -cv_scores

print("\nResultados de validación cruzada (RMSE):")
print(f"Media: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")
print(f"Scores individuales: {cv_scores}")










