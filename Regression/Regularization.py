import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, SCORERS
import seaborn as sns

df = pd.read_csv('Advertising.csv')
X = df.drop('sales', axis=1)
y = df['sales']

poly = PolynomialFeatures(degree=3, include_bias=False)
poly.fit(X)
poly_features = poly.transform(X)
X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.3, random_state=101)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# -------------- Ridge regression with Cross-Validation -----------------

ridge = RidgeCV(alphas=(0.1, 1, 10), cv=None, scoring='neg_mean_absolute_error')
ridge_fit = ridge.fit(X_train, y_train)
test_pred = ridge_fit.predict(X_test)
MAE = mean_absolute_error(y_test, test_pred)
RMSE = np.sqrt(mean_squared_error(y_test, test_pred))
print(ridge_fit.alpha_)
# print(test_pred)
print(MAE)
print(RMSE)


# -------------- Lasso regression with Cross-Validation -----------------
lasso = LassoCV(eps=0.1, n_alphas=100, alphas=None, cv=5, max_iter=100000)
lasso.fit(X_train, y_train)
test_pred_ls = lasso.predict(X_test)
MAE_ls = mean_absolute_error(y_test, test_pred_ls)
RMSE_ls = np.sqrt(mean_squared_error(y_test, test_pred_ls))
print(lasso.alpha_)
print(MAE_ls)
print(RMSE_ls)

# -------------- Elastic (L1 and L2) regression with Cross-Validation -----------------
# leaning the model from ridge 10% to 100% lasso.
elastic = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], eps=0.1, n_alphas=100, max_iter=100000)
elastic.fit(X_train, y_train)
test_pred_els = elastic.predict(X_test)
MAE_els = mean_absolute_error(y_test, test_pred_els)
RMSE_els = np.sqrt(mean_squared_error(y_test, test_pred_els))
print(elastic.l1_ratio_)
print(elastic.alpha_)
print(MAE_els)
print(RMSE_els)

