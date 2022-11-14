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

X_train, X_other, y_train, y_other = train_test_split(X, y, random_state=101, test_size=0.3)
X_val, X_test, y_val, y_test = train_test_split(X_other, y_other, random_state=101, test_size=0.5)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)

ridge = Ridge(alpha=1)
ridge_fit = ridge.fit(X_train, y_train)
y_pred_val = ridge_fit.predict(X_val)
# MAE = mean_absolute_error(y_test, y_pred)
# RMSE = np.sqrt(mean_squared_error(y_test, y_pred_val))
# r2 = r2_score(y_test, y_pred_val)

y_pred = ridge_fit.predict(X_test)
RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
# r2 = r2_score(y_test, y_pred_val)

print(RMSE)






