# Fraudulent Wine Detection Algorithm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, plot_confusion_matrix
from sklearn.pipeline import make_pipeline

# Building final model based on the Exploratory Data Analysis

df = pd.read_csv('wine_fraud.csv')
df['type'] = pd.get_dummies(df['type'], drop_first=True)

X = df.drop('quality', axis=1)
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=101, test_size=0.1)
scale = StandardScaler()
# X_train = scale.fit_transform(X_train)
# X_test = scale.transform(X_test)
svc = SVC()

# C = np.linspace(0, 5, 100)
# poly_deg = list(range(1, 4))
# param_grid = {'C': C, 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'gamma': ['scale', 'auto']}
# GSCV = GridSearchCV(svc, param_grid, cv=5, scoring='accuracy', verbose=2)
# GSCV.fit(X_train, y_train)
# print(GSCV.best_estimator_)
# print(GSCV.best_estimator_.get_params())

# According to below report the best value of C = 0.505 and kernel poly is used. Degree of poly is 3 and gamma to be
# scale.

# Now evaluating our model based on GridSearchCV Result

# svc = SVC(C=0.3, kernel='rbf', degree=3, class_weight='balanced', gamma='auto')
# pipe = make_pipeline(scale, svc)
# pipe.fit(X_train, y_train)
# y_preds = pipe.predict(X_test)

# plot_confusion_matrix(pipe, X_test, y_test)
# print(accuracy_score(y_test, y_preds))
# print(classification_report(y_test, y_preds))
# plt.show()

# IF we use the values of GridsearchCV our model in case of fraudulent wine detection its fails. The reason behind
# the failure in unbalance classes hence we need to introduce the class_weight concept Initially,
# assuming class_weight to be balanced once deep dive into hyperparameter tuning will improve the overall model
# working.
# Next steps are as follows:

# 1. Hyperparameter tuning of SVC
# 2. Change of algorithm
# 3. Move back to EDA and do feature engineering in more detail
# 4. Repeat step#1 after selecting best model

# Reworking on our parameters

# C = np.linspace(0, 5, 50)
# poly_deg = list(range(1, 4))
# param_grid = {'C': C, 'kernel': ['poly', 'rbf'], 'gamma': ['auto'], 'class_weight': ['balanced']}
# GSCV = GridSearchCV(svc, param_grid, cv=5, verbose=2)
# GSCV.fit(X_train, y_train)
# print(GSCV.best_estimator_)
# print(GSCV.best_estimator_.get_params())

# Rerunning our model once again
svc = SVC(C=0.115, kernel='poly', degree=3, class_weight='balanced', gamma='auto')
pipe = make_pipeline(scale, svc)
pipe.fit(X_train, y_train)
y_preds = pipe.predict(X_test)

plot_confusion_matrix(pipe, X_test, y_test, cmap='cividis')
print(classification_report(y_test, y_preds))
plt.show()

# Much better parameters this in comparison with previous one. We need to rethink on the problem or need to change the evaluation metrics


# OUTPUT_1 OF GRIDSEARCHCV

# SVC(C=0.5050505050505051, kernel='poly') {'C': 0.5050505050505051, 'break_ties': False, 'cache_size': 200,
# 'class_weight': None, 'coef0': 0.0, 'decision_function_shape': 'ovr', 'degree': 3, 'gamma': 'scale',
# 'kernel': 'poly', 'max_iter': -1, 'probability': False, 'random_state': None, 'shrinking': True, 'tol': 0.001,
# 'verbose': False}

# OUTPUT_2 OF GRIDSEARCHCV

# SVC(C=0.10204081632653061, class_weight='balanced', gamma='auto', kernel='poly') {'C': 0.10204081632653061,
# 'break_ties': False, 'cache_size': 200, 'class_weight': 'balanced', 'coef0': 0.0, 'decision_function_shape': 'ovr',
# 'degree': 3, 'gamma': 'auto', 'kernel': 'poly', 'max_iter': -1, 'probability': False, 'random_state': None,
# 'shrinking': True, 'tol': 0.001, 'verbose': False}


# Using SVM Following are the results

# Parameters: C = 0.8 and Kernel = rbf   --> f1-score = 0.26, accuracy = 0.84, Fraud: (Precision: 0.16 , Recall: 0.67)
# Parameters: C = 0.115 and Kernel = poly(3)   --> f1-score = 0.31, accuracy = 0.9, Fraud: (Precision: 0.21 , Recall: 0.56)