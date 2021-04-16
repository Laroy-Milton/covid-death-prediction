from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
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

    # Decision Tree Regression
    DTR_parameters = {'splitter': ('best', 'random'),
                      'max_depth': np.arange(1, 10),
                      'min_samples_split': np.arange(2, 10),
                      'min_samples_leaf': np.arange(1, 5)}

    # ---------------------------------------------------------
    # Random Forest Regression
    model_name = 'Random Forest Regression All'
    model = RandomForestRegressor()
    X, y = extractAllData()
    bm = bestModel(model, model_name, RFR_parameters, X, np.ravel(y), cv)
    feature_imp = topFeatures(bm, (X, y), num_features=10)

    # ---------------------------------------------------------
    # Ridge regression Specific
    model_name = 'Ridge regression Specific'
    model = Ridge(normalize=True)
    X, y = extractData()
    bestModel(model, model_name, ridge_parameters, X, y, cv)

    # ---------------------------------------------------------
    # Ridge Regression ALL
    model_name = 'Ridge Regression ALL'
    model = Ridge()
    X, y = extractAllData()
    bestModel(model, model_name, ridge_parameters, X, y, cv)

    # ---------------------------------------------------------
    # Ridge regression Top 10 Feature importance
    model_name = 'Ridge Regression Top 10'
    model = Ridge()
    X, y = extractAllData()
    X = X[feature_imp[:10].index]
    bestModel(model, model_name, ridge_parameters, X, y, cv)

    # ---------------------------------------------------------
    # KNN
    model_name = 'KNN All'
    model = KNeighborsRegressor()
    X, y = extractAllData()
    bestModel(model, model_name, KNN_parameters, X, y, cv)

    # ---------------------------------------------------------
    # Decision tree Regression All
    model_name = 'Decision Tree Regressor All'
    model = DecisionTreeRegressor()
    X, y = extractAllData()
    bestModel(model, model_name, DTR_parameters, X, y, cv)


main()
