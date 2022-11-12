import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('wine_fraud.csv')
print(df.info())

# Checking the variable to predict
print(df[['type', 'quality']])

# It is binary classification problem. Checking class distribution.
print(df['quality'].value_counts())
sns.histplot(data=df, x='quality')

# Legit: 6251 ________ Fraud: 246 ``` Since output variable is not uniformly distributed hence data scaling is
# mandatory ```

print(df['type'].value_counts())
sns.histplot(data=df, x='type')
# There are two types of wind "RED: 1599" and "White: 4898" - Use Label Encoding

# Checking how many red/white wines are fraud or legit
sns.displot(data=df, x='type', hue='quality', multiple='dodge')


def check_per_type_val(color, qual):
    val = (df['type'] == color) & (df['quality'] == qual)
    return val.sum()


print(check_per_type_val('red', 'Legit'))
print(check_per_type_val('white', 'Legit'))
print(check_per_type_val('red', 'Fraud'))
print(check_per_type_val('white', 'Fraud'))

# RED --> Legit = 1536, RED --> Fraud = 63, WHITE --> Legit = 4715, WHITE --> Fraud = 183

df_encoded = pd.get_dummies(data=df)  # One Hot encoder can't be used as it will it split the columns
print(df_encoded.info())

# Feature Selection
# Checking the correlation between the "Quality" and other features

df['quality_enc'] = df['quality'].apply(lambda x: 1 if x == 'Legit' else 0)
sns.heatmap(data=df.corr(), annot=True)
print(df.corr()['quality_enc'][:-1].sort_values())

# Further exploring/visualizing the relationship between features

sns.clustermap(data=df.corr()[:-1], cmap='viridis', annot=True)

# Observations

# 1. Volatile acidity have high negative correlation with "Quality". Means the Legit quality alcohol must have lower volatile acidity
# 2. Chlorides have -ve relation with "Quality" but intresting fact is that it is cluster with Sulphates (+ve coorelation with output feature)

# Exploring more
df_Legit = df[df['quality'] == 'Legit']
print(df_Legit.describe().transpose())

df_fraud = df[df['quality'] == 'Fraud']
print(df_fraud.describe().transpose())

# From above, we can see the difference between good quality and bad quality wines
# 1. voltaile acidity in Legit wine is less in comparison with Fraud wines (-ve relation)
# 2. Quantity of residual sugar in Legit wines need to be on higher side (+ve relation)
# 3. More alcohol will be in the Legit wines (+ve relation)
# 4. total sulfur dioxide in legit wines will be high (+ve relation)

# Most important feature which will impact the quality of wine

plt.show()
