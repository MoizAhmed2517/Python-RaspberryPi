import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

df = pd.read_csv('quikr_car.csv')
df_backup = df.copy()


df = df[df['year'].str.isnumeric()]
df['year'] = df['year'].astype(int)

df = df[df['Price'] != 'Ask For Price']
df['Price'] = df['Price'].str.replace(",", "").astype(int)

df['fuel_type'].fillna('Petrol', inplace=True)

df['kms_driven'] = df['kms_driven'].str.replace(" kms", "")
df = df[df['kms_driven'] != 'Petrol']
df['kms_driven'] = df['kms_driven'].str.replace(",", "").astype(int)

df['name'] = df['name'].str.split(" ").str[0:3].str.join(" ")

df = df[df['Price'] != 8500003]

df = df.reset_index(drop=True)

# df.to_csv('Cleaned_Data_Car.csv')
# sns.pairplot(data=df, diag_kind='kde', hue='fuel_type')
# plt.show()

# Applying model

X = df.drop(columns='Price')
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=661)

ohe = OneHotEncoder()
ohe.fit(X[['name', 'company', 'fuel_type']])
col_trans = make_column_transformer((OneHotEncoder(categories=ohe.categories_), ['name', 'company', 'fuel_type']), remainder='passthrough')
lr = LinearRegression()
pipe = make_pipeline(col_trans, lr)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

r2score = r2_score(y_test, y_pred)
MAE = mean_absolute_error(y_test, y_pred)
RMSE = np.sqrt(mean_squared_error(y_test, y_pred))

print(r2score)
print(MAE)
print(RMSE)


