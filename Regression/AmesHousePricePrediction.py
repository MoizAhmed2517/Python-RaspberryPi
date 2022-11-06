import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error

df = pd.read_csv('AMES_Final_DF.csv')
X = df.drop(columns='SalePrice')
y = df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=101, test_size=0.1)
scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)

ENM = ElasticNet()
param_grids = {'alpha': [0.1, 1, 100], 'l1_ratio': [0.1, 0.5, 0.7, 0.95, 0.99, 1]}
GS = GridSearchCV(estimator=ENM, param_grid=param_grids, scoring=['neg_mean_squared_error', 'r2'], cv=10, verbose=2,
                  refit='neg_mean_squared_error')
GS.fit(X_train, y_train)
print(GS.best_estimator_)

y_pred = GS.predict(X_test)
print(f'RMSE: $ {np.sqrt(mean_squared_error(y_test, y_pred))}')
print(f'MAE: $ {mean_absolute_error(y_test, y_pred)}')
