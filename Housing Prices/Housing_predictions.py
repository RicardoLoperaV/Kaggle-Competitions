# Primero importamos las bibliotecas necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

current_dir = os.path.dirname(os.path.abspath(__file__))

#Cargamos el train y test 
test_df = pd.read_csv(os.path.join(current_dir, 'test.csv'))
train_df = pd.read_csv(os.path.join(current_dir, 'train.csv'))

print(train_df.head(5))

#Como buena practica, separamos los datos en numericos y categoricos 
#Con el fin de procesarlos de manera diferente
#quitamos la columna SalePrice pues es la variable que queremos predecir
drop_cols = ['Id', 'SalePrice','MiscVal','LowQualFinSF','BsmtHalfBath', 'KitchenAbvGr','3SsnPorch','ScreenPorch','PoolArea','MoSold','YrSold','MSSubClass']
cols_numericas = train_df.select_dtypes(include=['int64', 'float64']).columns.drop(drop_cols)
cols_categoricas = train_df.select_dtypes(include=['object']).columns

# Extraemos los valores de las características, no solo los nombres de columnas
X_numerical = train_df[cols_numericas]
X_categorical = train_df[cols_categoricas]

# Codificamos las columnas categóricas
#usando un one hot encoder para codificar las columnas categoricas
categorical_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_categorical_encoded = categorical_encoder.fit_transform(X_categorical)


#dentro de las columnas numericas, hay valores nulos
# Por lo cual los reemplazamos con la media
imputer = SimpleImputer(strategy='mean')
X_numerical = imputer.fit_transform(X_numerical)

# Normalizamos las columnas numéricas
scaler = StandardScaler()
X_numerical_scaled = scaler.fit_transform(X_numerical)

Y_train = train_df['SalePrice']

#combinamos las columnas numericas y categoricas
X_Combined = np.concatenate([X_numerical_scaled, X_categorical_encoded], axis=1)

#para la evaluacion final, usaremos el conjunto de entrenamiento completo
#por lo cual usamos la siguiente linea para conservar el conjunto Y original
Y_train_val_final = Y_train.copy()

#Separamos los datos en train y validation para evitar el sobreajuste
X_train, X_val, Y_train, Y_val = train_test_split(X_Combined, Y_train, test_size=0.2, random_state=1)


print("\nNumerical features shape:", X_numerical.shape)
print("Categorical features shape:", X_categorical.shape)

#usamos un arbol de decision para predecir
#ponemos un random state para que el modelo sea reproducible
#Con el .fit entrenamos el modelo
model = DecisionTreeRegressor(random_state=1, criterion='friedman_mse', splitter='random', max_depth=10)
model.fit(X_train, Y_train)

# Evaluación en conjunto de entrenamiento
train_predictions = model.predict(X_train)
train_mse = mean_squared_error(Y_train, train_predictions)
print("\nError en datos de entrenamiento (RMSE):", np.sqrt(train_mse))

# Evaluación en conjunto de validación
val_predictions = model.predict(X_val)
val_mse = mean_squared_error(Y_val, val_predictions)
print("Error en datos de validación (RMSE):", np.sqrt(val_mse))

# Predicciones en el conjunto de test
#preprocesamor el conjunto de test, primero separamos las columnas numericas y categoricas
test_categorical = test_df[cols_categoricas]
test_numerical = test_df[cols_numericas]

test_numerical = imputer.transform(test_numerical)
test_numerical = scaler.transform(test_numerical)

test_categorical_encoded = categorical_encoder.transform(test_categorical)
test_df = np.concatenate([test_numerical, test_categorical_encoded], axis=1)
predic = model.predict(test_df)

#una vez entrenado el modelo y las predicciones hechas, podemos evaluar el modelo
#usamos el error cuadratico medio para evaluar el modelo
final_predictions = model.predict(X_Combined)  
mse = mean_squared_error(Y_train_val_final, final_predictions)

#imprimimos el error cuadratico medio
print("\nRoot Mean Squared Error:", np.sqrt(mse))

#Puntaje publico: 24400.13031











