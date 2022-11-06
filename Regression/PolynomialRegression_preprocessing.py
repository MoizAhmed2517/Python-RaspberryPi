import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error
from joblib import dump, load

from LinearRegression import campaign

df = pd.read_csv('Advertising.csv')

X = df.drop('sales', axis=1)
y = df['sales']

degree = np.arange(1, 11, 1)

RMSE_test = []
RMSE_train = []

for i in degree:
    poly = PolynomialFeatures(i, include_bias=False)
    poly.fit(X)
    poly_features = poly.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.3, random_state=101)
    L_r = LinearRegression().fit(X_train, y_train)

    train_pred = L_r.predict(X_train)
    test_pred = L_r.predict(X_test)

    RMSE_cal_train = np.sqrt(mean_squared_error(y_train, train_pred))
    RMSE_train.append(RMSE_cal_train)

    RMSE_cal_test = np.sqrt(mean_squared_error(y_test, test_pred))
    RMSE_test.append(RMSE_cal_test)

# print(RMSE_train)
# print(RMSE_test)
#
# plt.plot(degree[:5], RMSE_train[:5], label='TRAIN RMSE')
# plt.plot(degree[:5], RMSE_test[:5], label='TEST RMSE', color='r')
# plt.xlabel('Degree of Poly')
# plt.ylabel('RMSE')
# plt.legend()
# plt.show()
# test_residuals = y_test - pred
# sns.scatterplot(x=y_test, y=test_residuals)
# plt.axhline(y=0, color='red', ls='--')
# plt.show()

# ----- Saving Polynomial Linear Regression model -----------

# final_poly_converter = PolynomialFeatures(3, include_bias=False)
# final_poly_converter.fit(X)
# final_poly_features = final_poly_converter.transform(X)
# final_model = LinearRegression()
# final_model.fit(final_poly_features, y)
# dump(final_model, 'final_poly_sales_model.joblib')
# dump(final_poly_converter, 'final_poly_converter.joblib')


# ---- Loading Polynomial Linear Regression model --------------------------------

loaded_converter = load('final_poly_converter.joblib')
loaded_model = load('final_poly_sales_model.joblib')

campaign = [[149, 22, 12]]
transformed_converter = loaded_converter.fit_transform(campaign)
print(loaded_model.predict(transformed_converter))
