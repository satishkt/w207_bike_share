from __future__ import division
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, Normalizer, PolynomialFeatures
from sklearn.qda import QDA
from sklearn.svm import SVR


__author__ = 'Satish'
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn import cross_validation, linear_model
import matplotlib.pyplot as plt
from sklearn.lda import LDA
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import cross_val_score, KFold

import bikeStandardizeData

np.random.seed(0)

clf_linear = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('normalizer', Normalizer()),
    ('model', LinearRegression())
])

scores = []
cv, train_data, train_labels = bikeStandardizeData.readData()
for i, (train_index, test_index) in enumerate(cv):
    x_train, y_train = train_data.ix[train_index, 'season':], train_labels.ix[train_index, 'count']
    x_test, y_test = train_data.ix[test_index, 'season':], train_labels.ix[test_index, 'count']
    clf_linear.fit(x_train, y_train)
    predicted_values = clf_linear.predict(x_test)
    # print("Cross-Validated Error Loop Linear  {0}: {1}".format(i, bikeStandardizeData.rmsle(y_test, predicted_values)))
    scores.append(clf_linear.score(x_test, y_test))

# print scores
#print [score for score in scores]
print("Mean Score for linear model : {}".format(np.mean(scores)))

clf_lda = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('normalizer', Normalizer()),
    ('model', LDA())
])

for i, (train_index, test_index) in enumerate(cv):
    x_train, y_train = train_data.ix[train_index, 'season':], train_labels.ix[train_index, 'count']
    x_test, y_test = train_data.ix[test_index, 'season':], train_labels.ix[test_index, 'count']
    clf_lda.fit(x_train, y_train)
    predicted_values = clf_lda.predict(x_test)
    #print("Cross-Validated Error Loop LDA  {0}: {1}".format(i, bikeStandardizeData.rmsle(y_test, predicted_values)))
    scores.append(clf_lda.score(x_test, y_test))

#print scores
#print [score for score in scores]
print("Mean Score for LDA model : {}".format(np.mean(scores)))

clf_knn = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('normalizer', Normalizer()),
    ('model', KNeighborsRegressor(n_neighbors=5))
])

for i, (train_index, test_index) in enumerate(cv):
    x_train, y_train = train_data.ix[train_index, 'season':], train_labels.ix[train_index, 'count']
    x_test, y_test = train_data.ix[test_index, 'season':], train_labels.ix[test_index, 'count']
    clf_knn.fit(x_train, y_train)
    predicted_values = clf_knn.predict(x_test)
    #print("Cross-Validated Error Loop KNN  {0}: {1}".format(i, bikeStandardizeData.rmsle(y_test, predicted_values)))
    scores.append(clf_knn.score(x_test, y_test))

#print scores
#print [score for score in scores]
print("Mean Score for KNN model : {}".format(np.mean(scores)))

clf_poly = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('normalizer', Normalizer()),
    ('poly', PolynomialFeatures(degree=3)),
    ('model', LinearRegression())
])

for i, (train_index, test_index) in enumerate(cv):
    x_train, y_train = train_data.ix[train_index, 'season':], train_labels.ix[train_index, 'count']
    x_test, y_test = train_data.ix[test_index, 'season':], train_labels.ix[test_index, 'count']
    clf_poly.fit(x_train, y_train)
    predicted_values = clf_poly.predict(x_test)
    #print("Cross-Validated Error Loop KNN  {0}: {1}".format(i, bikeStandardizeData.rmsle(y_test, predicted_values)))
    scores.append(clf_poly.score(x_test, y_test))

#print scores
#print [score for score in scores]
print("Mean Score for Polynomial  model : {}".format(np.mean(scores)))

clf_ridge = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('normalizer', Normalizer()),
    ('poly', PolynomialFeatures(degree=3)),
    ('model', Ridge(alpha=0.5, copy_X=True, fit_intercept=True, max_iter=None,
                    normalize=False, solver='auto', tol=0.001))
])

for i, (train_index, test_index) in enumerate(cv):
    x_train, y_train = train_data.ix[train_index, 'season':], train_labels.ix[train_index, 'count']
    x_test, y_test = train_data.ix[test_index, 'season':], train_labels.ix[test_index, 'count']
    clf_ridge.fit(x_train, y_train)
    predicted_values = clf_ridge.predict(x_test)
    #print("Cross-Validated Error Loop KNN  {0}: {1}".format(i, bikeStandardizeData.rmsle(y_test, predicted_values)))
    scores.append(clf_ridge.score(x_test, y_test))

#print scores
#print [score for score in scores]
print("Mean Score for Ridge  model : {}".format(np.mean(scores)))

clf_lasso = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('normalizer', Normalizer()),
    ('poly', PolynomialFeatures(degree=3)),
    ('model', Lasso(alpha=0.1, copy_X=True, fit_intercept=True, max_iter=1000,
                    normalize=False, positive=False, precompute=False,
                    tol=0.0001, warm_start=False))
])

for i, (train_index, test_index) in enumerate(cv):
    x_train, y_train = train_data.ix[train_index, 'season':], train_labels.ix[train_index, 'count']
    x_test, y_test = train_data.ix[test_index, 'season':], train_labels.ix[test_index, 'count']
    clf_lasso.fit(x_train, y_train)
    predicted_values = clf_lasso.predict(x_test)
    #print("Cross-Validated Error Loop KNN  {0}: {1}".format(i, bikeStandardizeData.rmsle(y_test, predicted_values)))
    scores.append(clf_lasso.score(x_test, y_test))

#print scores
#print [score for score in scores]
print("Mean Score for Lasso  model : {}".format(np.mean(scores)))



# Fit regression model

clf_svrbf = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('normalizer', Normalizer()),
    ('model', SVR(kernel='rbf', C=1e3, gamma=0.1))
])
for i, (train_index, test_index) in enumerate(cv):
    x_train, y_train = train_data.ix[train_index, 'season':], train_labels.ix[train_index, 'count']
    x_test, y_test = train_data.ix[test_index, 'season':], train_labels.ix[test_index, 'count']
    clf_svrbf.fit(x_train, y_train)
    predicted_values = clf_svrbf.predict(x_test)
    #print("Cross-Validated Error Loop KNN  {0}: {1}".format(i, bikeStandardizeData.rmsle(y_test, predicted_values)))
    scores.append(clf_svrbf.score(x_test, y_test))

#print scores
#print [score for score in scores]
print("Mean Score for clf_svrbf : {}".format(np.mean(scores)))

clf_svrlin = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('normalizer', Normalizer()),
    ('model', SVR(kernel='linear', C=1e3))
])

for i, (train_index, test_index) in enumerate(cv):
    x_train, y_train = train_data.ix[train_index, 'season':], train_labels.ix[train_index, 'count']
    x_test, y_test = train_data.ix[test_index, 'season':], train_labels.ix[test_index, 'count']
    clf_svrlin.fit(x_train, y_train)
    predicted_values = clf_svrlin.predict(x_test)
    #print("Cross-Validated Error Loop KNN  {0}: {1}".format(i, bikeStandardizeData.rmsle(y_test, predicted_values)))
    scores.append(clf_svrlin.score(x_test, y_test))

#print scores
#print [score for score in scores]
print("Mean Score for clf_svrlin : {}".format(np.mean(scores)))

clf_svrpoly = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('normalizer', Normalizer()),
    ('model', SVR(kernel='poly', C=1e3, degree=3))
])

for i, (train_index, test_index) in enumerate(cv):
    x_train, y_train = train_data.ix[train_index, 'season':], train_labels.ix[train_index, 'count']
    x_test, y_test = train_data.ix[test_index, 'season':], train_labels.ix[test_index, 'count']
    clf_svrpoly.fit(x_train, y_train)
    predicted_values = clf_svrpoly.predict(x_test)
    #print("Cross-Validated Error Loop KNN  {0}: {1}".format(i, bikeStandardizeData.rmsle(y_test, predicted_values)))
    scores.append(clf_svrpoly.score(x_test, y_test))

#print scores
#print [score for score in scores]
print("Mean Score for clf_svrpoly : {}".format(np.mean(scores)))

rf = RandomForestRegressor(random_state=0, n_estimators=100)
clf_rf = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('normalizer', Normalizer()),
    ('model', rf)
])

for i, (train_index, test_index) in enumerate(cv):
    x_train, y_train = train_data.ix[train_index, 'season':], train_labels.ix[train_index, 'count']
    x_test, y_test = train_data.ix[test_index, 'season':], train_labels.ix[test_index, 'count']
    clf_rf.fit(x_train, y_train)
    predicted_values = clf_rf.predict(x_test)
    #print("Cross-Validated Error Loop KNN  {0}: {1}".format(i, bikeStandardizeData.rmsle(y_test, predicted_values)))
    scores.append(clf_rf.score(x_test, y_test))

#print scores
#print [score for score in scores]
print("Mean Score for clf_rf : {}".format(np.mean(scores)))

gbm = GradientBoostingRegressor(loss='ls', alpha=0.95, n_estimators=500, max_depth=4, learning_rate=.01,
                                min_samples_leaf=9, min_samples_split=9)

clf_gbm = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('normalizer', Normalizer()),
    ('model', gbm)
])

for i, (train_index, test_index) in enumerate(cv):
    x_train, y_train = train_data.ix[train_index, 'season':], train_labels.ix[train_index, 'count']
    x_test, y_test = train_data.ix[test_index, 'season':], train_labels.ix[test_index, 'count']
    clf_gbm.fit(x_train, y_train)
    predicted_values = clf_gbm.predict(x_test)
    #print("Cross-Validated Error Loop KNN  {0}: {1}".format(i, bikeStandardizeData.rmsle(y_test, predicted_values)))
    scores.append(clf_gbm.score(x_test, y_test))

#print scores
#print [score for score in scores]
print("Mean Score for clf_gbm : {}".format(np.mean(scores)))

