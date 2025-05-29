# Primero importamos las bibliotecas necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

current_dir = os.path.dirname(os.path.abspath(__file__))

#Cargamos el train y test 
test_df = pd.read_csv(os.path.join(current_dir, 'test.csv'))
train_df = pd.read_csv(os.path.join(current_dir, 'train.csv'))

print(train_df.head(5))

#Como buena practica, separamos los datos en numericos y categoricos 
#Con el fin de procesarlos de manera diferente
#quitamos la columna SalePrice pues es la variable que queremos predecir
drop_cols = ['Id', 'SalePrice']
cols_numericas = train_df.select_dtypes(include=['int64', 'float64']).columns.drop(drop_cols)
cols_categoricas = train_df.select_dtypes(include=['object']).columns

# Extraemos los valores de las características, no solo los nombres de columnas
X_numerical = train_df[cols_numericas]
X_categorical = train_df[cols_categoricas]

Y_train = train_df['SalePrice']

print("\nNumerical features shape:", X_numerical.shape)
print("Categorical features shape:", X_categorical.shape)

#usamos un arbol de decision para predecir
#ponemos un random state para que el modelo sea reproducible
model = DecisionTreeRegressor(random_state=1)
model.fit(X_numerical, Y_train)

#Con el .fit entrenamos el modelo

#hacemos predicciones
predic = model.predict(test_df[cols_numericas])

#una vez entrenado el modelo y las predicciones hechas, podemos evaluar el modelo
#usamos el error cuadratico medio para evaluar el modelo
train_predictions = model.predict(X_numerical)  
mse = mean_squared_error(Y_train, train_predictions)

#imprimimos el error cuadratico medio
print("\nRoot Mean Squared Error:", np.sqrt(mse))













