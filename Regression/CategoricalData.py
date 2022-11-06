import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Ames_no_missingdata.csv')
df['MS SubClass'] = df['MS SubClass'].apply(str)

my_object_df = df.select_dtypes(include='object')
my_numeric_df = df.select_dtypes(exclude='object')

df_object_dummies = pd.get_dummies(my_object_df, drop_first=True)

final_df = pd.concat([my_numeric_df, df_object_dummies], axis=1)

print(final_df.columns)