import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
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

rmse_lst = []
lst = []
grid = np.linspace(0, 1, 100)
for i in grid:
    ridge = Ridge(alpha=i)
    ridge_fit = ridge.fit(X_train, y_train)
    y_pred = ridge_fit.predict(X_test)
    # MAE = mean_absolute_error(y_test, y_pred)
    RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    lst.append(r2)
    rmse_lst.append(RMSE)

# print(RMSE)
# print(r2)
print(max(lst))
print(lst.index(max(lst)))
print(max(rmse_lst))
print(rmse_lst.index(min(rmse_lst)))






