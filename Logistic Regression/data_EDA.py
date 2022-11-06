import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve

df = pd.read_csv('hearing_test.csv')
# print(df.describe())
# print(df['test_result'].value_counts())
#
# # sns.countplot(data=df, x='test_result')
# # sns.boxplot(data=df, x='test_result', y='physical_score')
# # sns.scatterplot(data=df, x='age', y='physical_score', hue='test_result')
# sns.heatmap(df.corr(), annot=True)
#
# # using 3d plot in case of classification
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(df['age'], df['physical_score'], df['test_result'], c=df['test_result'])
#
# plt.show()

# Training Model

X = df.drop('test_result', axis=1)
y = df['test_result']

X_test, X_train, y_test, y_train = train_test_split(X, y, random_state=101, test_size=0.1)
scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)

logic = LogisticRegression()
logic.fit(X_train, y_train)
print(logic.coef_)

y_pred = logic.predict(X_test)
y_log_prob = logic.predict_log_proba(X_test)
y_prob = logic.predict_proba(X_test)


print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
plot_confusion_matrix(logic, X_test, y_test, normalize='all')


print(classification_report(y_test, y_pred))

plot_roc_curve(logic, X_test, y_test)
plot_precision_recall_curve(logic, X_test, y_test)
plt.show()

