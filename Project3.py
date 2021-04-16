import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn


from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import Normalizer, MinMaxScaler
from sklearn.tree import DecisionTreeRegressor

from utils import *

np.random.seed(32455)
cv = 5  # k fold


def main():
    RFR_parameters = {'n_estimators': np.arange(2, 5),
                      'criterion': ['mse', 'mae'],
                      'min_samples_split': np.linspace(0.1, 1, num=10),
                      'max_features': ['auto', 'sqrt', 'log2'],
                      'bootstrap': [True, False],
                      'n_jobs': [-1]}

    ridge_parameters = {'alpha': np.linspace(0.0, 1.0, num=30),
                        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
                        'max_iter': [1000, 10000, 50000]}

    KNN_parameters = {'n_neighbors': [30],  # np.arange(2,10),
                      'weights': ['uniform', 'distance'],
                      'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                      'leaf_size': np.arange(2, 30),
                      'p': [1, 2],
                      'n_jobs': [-1]}

    DTR_parameters = {'splitter': ('best', 'random'),
                      'max_depth': np.arange(1, 10),
                      'min_samples_split': np.arange(2, 10),
                      'min_samples_leaf': np.arange(1, 5)}

    # ---------------------------------------------------------
    # Random Forest Regression
    model_name = 'Random Forest Regression All'
    print('Training ' + model_name + '...')

    X, y = extractAllData()

    RFR = RandomForestRegressor()

    gs = GridSearchCV(RFR, RFR_parameters, cv=cv, scoring='r2', return_train_score=True)
    gs.fit(X, np.ravel(y))

    feature_imp = pd.Series(gs.best_estimator_.feature_importances_, index=list(X)).sort_values(ascending=False)
    sn.barplot(x=feature_imp[:10], y=feature_imp[:10].index)
    plt.title('Top 10 important features')
    plt.tight_layout()
    print("Top 10 features:")
    print(feature_imp[:10])
    plt.show()

    title = '{} {:.2}'.format(model_name, gs.best_score_)
    print(title)
    plot_learning_curve(gs.best_estimator_, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5))

    # ---------------------------------------------------------
    # Ridge regression Specific
    model_name = 'Ridge regression Specific'
    print('Training ' + model_name + '...')

    X, y = extractData()

    reg = Ridge(normalize=True)

    gs = GridSearchCV(reg, ridge_parameters, cv=cv, scoring='r2', return_train_score=True)
    gs.fit(X, y)

    plotLearningCurve(gs, 'Ridge regression Specifc')

    title = '{} {:.2}'.format(model_name, gs.best_score_)
    print(title)
    plot_learning_curve(gs.best_estimator_, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5))

    # ---------------------------------------------------------
    # Ridge Regression ALL
    model_name = 'Ridge Regression ALL'
    print('Training ' + model_name + '...')

    X, y = extractAllData()

    reg = Ridge()

    gs = GridSearchCV(reg, ridge_parameters, cv=cv, scoring='r2', return_train_score=True)
    gs.fit(X, y)

    title = '{} {:.2}'.format(model_name, gs.best_score_)
    print(title)
    plot_learning_curve(gs.best_estimator_, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5))

    # ---------------------------------------------------------
    # Ridge regression Top 10 Feature importance
    model_name = 'Ridge Regression Top 10'
    print('Training ' + model_name + '...')

    X, y = extractAllData()
    X = X[feature_imp[:10].index]

    reg = Ridge()

    gs = GridSearchCV(reg, ridge_parameters, cv=cv, scoring='r2', return_train_score=True)
    gs.fit(X, y)

    title = '{} {:.2}'.format(model_name, gs.best_score_)
    print(title)
    plot_learning_curve(gs.best_estimator_, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5))

    # ---------------------------------------------------------
    # KNN
    model_name = 'KNN All'
    print('Training ' + model_name + '...')
    X, y = extractAllData()

    knn = KNeighborsRegressor()

    gs = GridSearchCV(knn, KNN_parameters, scoring='r2', return_train_score=True)
    gs.fit(X, y)

    title = '{} {:.2}'.format(model_name, gs.best_score_)
    print(title)
    title = '{} {:.2}'.format(model_name, gs.best_score_)
    plot_learning_curve(gs.best_estimator_, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5))

    # ---------------------------------------------------------
    # Decision tree Regression All
    model_name = 'Decision Tree Regressor All'
    print('Training ' + model_name + '...')



    gs = GridSearchCV(DecisionTreeRegressor(), DTR_parameters, cv=cv, scoring='r2', return_train_score=True)
    gs.fit(X, y)

    title = '{} {:.2}'.format(model_name, gs.best_score_)
    print(title)
    plot_learning_curve(gs.best_estimator_, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5))
    model = DecisionTreeRegressor()
    data = extractAllData()
    trainModel(model, model_name, DTR_parameters, data)


def trainModel(model, model_name, params, data):
    print('Training ' + model_name + '...')
    X, y = data

    gs = GridSearchCV(model, params, cv=cv, scoring='r2', return_train_score=True)
    gs.fit(X, y)

    title = '{} {:.2}'.format(model_name, gs.best_score_)
    print(title)
    plot_learning_curve(gs.best_estimator_, title, X, y)

main()
