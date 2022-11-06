import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

df = pd.read_csv('Advertising.csv')

X = df.drop(columns='sales')
y = df['sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=101, test_size=0.3)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model_ENM = ElasticNet()
param_grid = {'alpha': [0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15], 'l1_ratio': [0.1, 0.5, 0.7, 0.95, 0.99, 1]}
GScv = GridSearchCV(estimator=model_ENM, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, verbose=2, refit='neg_mean_squared_error')
GScv.fit(X_train, y_train)
print(GScv.best_estimator_)

y_pred = GScv.predict(X_test)
print(mean_squared_error(y_test, y_pred))

