# Primero importamos las bibliotecas necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.tree import DecisionTreeRegressor

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

#Cargamos el train y test 
test_df = pd.read_csv(os.path.join(current_dir, 'test.csv'))
train_df = pd.read_csv(os.path.join(current_dir, 'train.csv'))


print(train_df.head(5))

#Como buena practica, separamos los datos en numericos y categoricos 
#Con el fin de procesarlos de manera diferente
#quitamos la columna SalePrice pues es la variable que queremos predecir
drop_numerical_cols = ['Id', 'SalePrice']
X_numerical = train_df.select_dtypes(include = ['int64', 'float64']).columns.drop(drop_numerical_cols)
X_categorical = train_df.select_dtypes(include = ['object'])

Y_train = train_df['SalePrice']

#print(X_numerical.head(5))
print(X_categorical.head(5))

model = DecisionTreeRegressor()
model.fit(X_numerical, Y_train)






