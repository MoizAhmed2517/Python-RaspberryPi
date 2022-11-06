import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# df = pd.read_csv('Ames_Housing_Data.csv')
# print(df.corr()['SalePrice'].sort_values())
#
#
# print(df[(df['Overall Qual'] > 8) & (df['SalePrice'] < 200000)])
# drop_index = df[(df['Gr Liv Area'] > 4000) & (df['SalePrice'] < 400000)].index
# df = df.drop(drop_index, axis=0)
#
# # sns.scatterplot(data=df, x='Overall Qual', y='SalePrice')
# sns.scatterplot(data=df, x='Gr Liv Area', y='SalePrice')
# plt.show()
#
# df.to_csv('Ames_Housing_data_outlier_rm.csv')

df = pd.read_csv('Ames_outliers_removed.csv')
# print(df.info())
# print(df.head)
df = df.drop("PID", axis=1)

def isNullMissingPer(data):
    per_nan = 100 * data.isnull().sum() / len(df)
    per_nan = per_nan[per_nan > 0].sort_values()

    return per_nan

# sns.barplot(x=per_nan.index, y=per_nan)
# plt.xticks(rotation=90)
# plt.show()

df = df.dropna(axis=0, subset=['Electrical', 'Garage Cars'])


# print(df[df['Bsmt Half Bath'].isnull()])
# print(df[df['Bsmt Full Bath'].isnull()])

bsmt_num_cols = ['Bsmt Unf SF', 'Total Bsmt SF', 'BsmtFin SF 2', 'BsmtFin SF 1', 'Bsmt Full Bath', 'Bsmt Half Bath']
df[bsmt_num_cols] = df[bsmt_num_cols].fillna(0)

bsmt_str_cols = ['Bsmt Qual', 'Bsmt Cond', 'BsmtFin Type 1', 'BsmtFin Type 2', 'Bsmt Exposure']
df[bsmt_str_cols] = df[bsmt_str_cols].fillna('None')

# print(df[df['Mas Vnr Type'].isnull()])
# print(df[df['Mas Vnr Area'].isnull()])

df['Mas Vnr Type'] = df['Mas Vnr Type'].fillna('None')
df['Mas Vnr Area'] = df['Mas Vnr Area'].fillna(0)

Gar_str_cols = ['Garage Type', 'Garage Finish', 'Garage Qual', 'Garage Cond']
df[Gar_str_cols] = df[Gar_str_cols].fillna('None')
df['Garage Yr Blt'] = df['Garage Yr Blt'].fillna(0)

df = df.drop(['Alley', 'Pool QC', 'Misc Feature', 'Fence'], axis=1)
df['Fireplace Qu'] = df['Fireplace Qu'].fillna('None')

df['Lot Frontage'] = df.groupby('Neighborhood')['Lot Frontage'].transform(lambda value: value.fillna(value.mean()))
df['Lot Frontage'] = df['Lot Frontage'].fillna(0)

# per_nan = isNullMissingPer(df)
# print(per_nan)

df.to_csv('Ames_no_missingdata.csv')
