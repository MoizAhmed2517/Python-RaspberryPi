import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import scipy as sp
from joblib import dump, load

# Q: What is the relationship between each advertising channel (TV, Radio, Newspaper) and sales

df = pd.read_csv('Advertising.csv')

# sns.pairplot(df)
# plt.show()
X = df.drop('sales', axis=1)
y = df['sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
L_r = LinearRegression().fit(X_train, y_train)
pred = L_r.predict(X_test)

# print(df['sales'].mean())
# sns.histplot(x=pred, color='red')
# sns.histplot(x=y_test)
# plt.show()

# print(mean_absolute_error(y_test, pred))
# print(np.sqrt(mean_squared_error(y_test, pred)))

test_residuals = y_test - pred
# sns.scatterplot(x=y_test, y=test_residuals)
# plt.axhline(y=0, color='red', ls='--')
# sns.displot(test_residuals, bins=25, kde=True)
# fig, ax = plt.subplots(figsize=(6,8), dpi=100)
# _ = sp.stats.probplot(test_residuals, plot=ax)


# Model Deployment

final_model = LinearRegression()
final_model.fit(X, y)
# print(final_model.coef_)

y_pred = final_model.predict(X)

# fig, ax = plt.subplots(1, 3, figsize=(16,6))
#
# ax[0].plot(df['TV'], df['sales'], 'o')
# ax[0].plot(df['TV'], y_pred, 'o', color='r')
# ax[0].set_ylabel('Sales')
# ax[0].set_title('TV Spend')
#
# ax[1].plot(df['radio'], df['sales'], 'o')
# ax[1].plot(df['radio'], y_pred, 'o', color='r')
# ax[1].set_ylabel('Sales')
# ax[1].set_title('Radio Spend')
#
# ax[2].plot(df['newspaper'], df['sales'], 'o')
# ax[2].plot(df['newspaper'], y_pred, 'o', color='r')
# ax[2].set_ylabel('Sales')
# ax[2].set_title('Newspaper Spend')
#
# plt.show()

# dump(final_model, 'final_sales_model.joblib')

loaded_sales_model = load('final_sales_model.joblib')
# 149 TV, 22 Radio, 12 Newspaper, What would be the expected sales?
campaign = [[149, 22, 12]]
print(loaded_sales_model.predict(campaign))

