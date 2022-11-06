import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv('Advertising.csv')
df['spent'] = df['TV'] + df['radio'] + df['newspaper']

X = df['spent']
y = df['sales']

Coeff = np.polyfit(X, y, deg=1)
potential_spend = np.linspace(0, 500, 100)
predicted_sales = Coeff[0] * potential_spend + Coeff[1]

# sns.scatterplot(x=df['spent'], y=df['sales'])
# plt.plot(potential_spend, predicted_sales, color='red')
# plt.show()

# What will be the predicted sales if we spend 200 on media?

# spend = 200
# predicted_sales = Coeff[0] * spend + Coeff[1]
# print(predicted_sales)

# Changing the polunomial degree

Coeffthird = np.polyfit(X, y, 3)
pred_sales = Coeffthird[0] * (potential_spend ** 3) + Coeffthird[1] * potential_spend ** 2 + Coeffthird[2] * potential_spend + Coeffthird[3]

sns.scatterplot(x=df['spent'], y=df['sales'])
plt.plot(potential_spend, pred_sales, color='red')
plt.show()