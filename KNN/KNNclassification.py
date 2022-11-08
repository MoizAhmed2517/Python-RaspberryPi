import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.pipeline import Pipeline

df = pd.read_csv('gene_expression.csv')
# print(df.info())
# print(df.describe())
# print(df['Cancer Present'].value_counts())

# sns.displot(data=df, x='Cancer Present')
# plt.figure("Figure 2")
# sns.histplot(data=df, x='Gene One')
# plt.figure("Figure 3")
# sns.boxplot(data=df, x='Cancer Present', y='Gene One')
# plt.figure("Figure 4")
# sns.boxplot(data=df, x='Cancer Present', y='Gene Two')
# plt.figure("Figure 5")
# sns.pairplot(data=df, hue='Cancer Present')
# plt.show()
test_error_rates = []
X = df.drop('Cancer Present', axis=1)
y = df['Cancer Present']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=101, test_size=0.3)
# scale = StandardScaler()
# X_train = scale.fit_transform(X_train)
# X_test = scale.transform(X_test)
#
# for k in range(1, 30):
#     KNN = KNeighborsClassifier(n_neighbors=k)
#     KNN.fit(X_train, y_train)
#     y_pred = KNN.predict(X_test)
#     test_error_rates.append(1-(accuracy_score(y_test, y_pred)))
#
# plt.plot(test_error_rates)
# # plt.ylim([0.055, 0.070])
# plt.show()
#
# Knn = KNeighborsClassifier(n_neighbors=5)
# Knn.fit(X_train, y_train)
# y_pred_final = Knn.predict(X_test)
#
# print(confusion_matrix(y_test, y_pred_final))
# print(classification_report(y_test, y_pred_final))

# Setting Pipeline and making Grid Search CV

scaler = StandardScaler()
knn = KNeighborsClassifier()
operations = [('scaler', scaler), ('knn', knn)]

pipe = Pipeline(operations)


k_val = list(range(1, 15))
param_grid = {'knn__n_neighbors': k_val}
GSCV = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')
GSCV.fit(X_train, y_train)

print(GSCV.best_estimator_.get_params())

y_pred_full = GSCV.predict(X_test)

print(classification_report(y_test, y_pred_full))

new_patient = [[2.9, 1.4]]
print(GSCV.predict(new_patient))
print(GSCV.predict_proba(new_patient))

