from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.preprocessing import PowerTransformer, MinMaxScaler, RobustScaler, PolynomialFeatures
from sklearn.preprocessing import QuantileTransformer

from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import FunctionTransformer

from utils import *

np.random.seed(42)
cv = 5  # k fold


def main():
    RFR_parameters = {'n_estimators': np.arange(2, 5),
                      'criterion': ['mse', 'mae'],
                      'min_samples_split': np.linspace(0.1, 1, num=10),
                      'max_features': ['auto', 'sqrt', 'log2'],
                      'bootstrap': [True, False],
                      'n_jobs': [-1]}

    sgd_parameters = {'loss': ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
                      'penalty': ['l2', 'l1', 'elasticnet'],
                      'alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 1.5, 5, 10.0, 15],
                      'max_iter': [10000, 100000],
                      'shuffle': [True, False],
                      'epsilon': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0],
                      'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
                      'n_iter_no_change': np.arange(5, 20, 2)}

    ridge_parameters = {'alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 1.5, 5, 10.0, 15],
                        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
                        'max_iter': [800, 1000, 5000, 10000, 15000, 50000]}

    KNN_parameters = {'n_neighbors': [30],  # np.arange(2,10),
                      'weights': ['uniform', 'distance'],
                      'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                      'leaf_size': np.arange(2, 30),
                      'p': [1, 2],
                      'n_jobs': [-1]}

    # Decision Tree Regression
    DTR_parameters = {'splitter': ['best', 'random'],
                      'max_depth': np.arange(1, 10),
                      'min_samples_split': np.arange(2, 40, 5),
                      'min_samples_leaf': np.arange(2, 20, 2),
                      'min_weight_fraction_leaf': [0, 0.5],
                      'max_features': ['auto', 'sqrt', 'log2']}

    GPR_parameters = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
                      'n_restarts_optimizer': np.arange(0, 5)}

    GBR_parameters = {'loss': ['ls', 'lad', 'huber', 'quantile'],
                      'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
                      'n_estimators': [50, 100, 200, 300],
                      'criterion': ['friedman_mse', 'mse', 'mae'],
                      'min_samples_split': np.arange(2, 40, 5),
                      'min_samples_leaf': np.arange(2, 20, 2),
                      'min_weight_fraction_leaf': [0, 0.5],
                      'max_depth': np.arange(1, 10),
                      'max_features': ['auto', 'sqrt', 'log2'],
                      'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]}

    # ---------------------------------------------------------
    # Data Vis
    X, y = extractAllData()

    bc = PowerTransformer(method='box-cox', standardize=True)
    yj = PowerTransformer(method='yeo-johnson', standardize=True)
    qt = QuantileTransformer(n_quantiles=len(X.index)-30, output_distribution='normal') # prone to over fitting
    qtu = QuantileTransformer(n_quantiles=len(X.index)-30, output_distribution='uniform')

    norm = Normalizer(norm='max').fit(X)


    plt.hist(X)
    plt.title('No normalization')
    plt.show()

    Xsc = StandardScaler().fit_transform(X)
    plt.hist(Xsc)
    plt.title('Standard Scaler')
    plt.show()

    XminMax = MinMaxScaler().fit_transform(X)
    plt.hist(XminMax)
    plt.title('Min Max')
    plt.show()

    Xrs = RobustScaler().fit_transform(X)
    plt.hist(Xrs)
    plt.title('Robust Scaler')
    plt.show()

    # Xbc = bc.fit_transform(X)
    # plt.hist(Xbc)
    # plt.title('box-cox')
    # plt.show()

    Xyj = yj.fit_transform(X)
    plt.title('yeo-johnson')
    plt.hist(Xyj)
    plt.show()

    Xqt = qt.fit_transform(X)
    plt.hist(Xqt)
    plt.title('Quantile normal')
    plt.show()

    Xqtu = qtu.fit_transform(X)
    plt.hist(Xqtu)
    plt.title('Quantile uniform')
    plt.show()


    # ---------------------------------------------------------
    # Random Forest Regression
    model_name = 'Random Forest Regression All'
    model = RandomForestRegressor()

    X, y = extractAllData()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

    bm = bestModel(model, model_name, RFR_parameters, X_train, X_test, np.ravel(y_train), np.ravel(y_test))
    feature_imp = topFeatures(bm, (X_train, y_train), num_features=10)

    # # ---------------------------------------------------------
    # # SGD All
    # model_name = 'SGD Regressor All'
    # model = SGDRegressor()
    # X, y = extractAllData()
    # bestModel(model, model_name, sgd_parameters, X, np.ravel(y), cv)
    #
    # # ---------------------------------------------------------
    # # SGD Specific
    # model_name = 'SGD Regressor Specific'
    # model = SGDRegressor()
    # X, y = extractData()
    # bestModel(model, model_name, sgd_parameters, X, np.ravel(y), cv)
    #
    # # ---------------------------------------------------------
    # # SGD Top
    # model_name = 'SGD Regressor Top10 '
    # model = SGDRegressor()
    # X, y = extractAllData()
    # X = X[feature_imp[:10].index]
    # # bestModel(model, model_name, sgd_parameters, X, np.ravel(y), cv)
    #
    # # ---------------------------------------------------------
    # # Ridge Regression ALL
    # model_name = 'Ridge Regression ALL'
    # model = Ridge()
    # X, y = extractAllData()
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    # bestModel(model, model_name, ridge_parameters, X_train, X_test, y_train, y_test)
    #
    # # ---------------------------------------------------------
    # # Ridge regression Specific
    # model_name = 'Ridge regression Specific'
    # model = Ridge()
    # X, y = extractData()
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    # bestModel(model, model_name, ridge_parameters, X_train, X_test, y_train, y_test)
    #
    # # ---------------------------------------------------------
    # # Ridge regression Top 10 Feature importance
    # model_name = 'Ridge Regression Top 10'
    # model = Ridge()
    # X, y = extractAllData()
    # X = X[feature_imp[:10].index]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    # bestModel(model, model_name, ridge_parameters, X_train, X_test, y_train, y_test)

    # ---------------------------------------------------------
    # # KNN
    # model_name = 'KNN All'
    # model = KNeighborsRegressor()
    # X, y = extractAllData()
    # bestModel(model, model_name, KNN_parameters, X, y)

    # ---------------------------------------------------------
    # Decision tree Regression All
    model_name = 'Decision Tree Regressor All'
    model = DecisionTreeRegressor()
    X, y = extractAllData()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    bestModel(model, model_name, DTR_parameters, X_train, X_test, y_train, y_test)

    # ---------------------------------------------------------
    # Decision tree Regression Specific
    model_name = 'Decision Tree Regressor Specific'
    model = DecisionTreeRegressor()
    X, y = extractData()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    bestModel(model, model_name, DTR_parameters, X_train, X_test, y_train, y_test)

    # ---------------------------------------------------------
    # Decision tree Regression Top 10 Feature importance
    model_name = 'Decision Tree Regressor Top 10'
    model = DecisionTreeRegressor()
    X, y = extractAllData()
    X = X[feature_imp[:10].index]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    bestModel(model, model_name, DTR_parameters, X_train, X_test, y_train, y_test)

    # ---------------------------------------------------------
    # Gaussian Process Regressor All
    model_name = 'Gaussian Process Regressor All'
    model = GaussianProcessRegressor()
    X, y = extractAllData()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    bestModel(model, model_name, GPR_parameters, X_train, X_test, y_train, y_test)

    # ---------------------------------------------------------
    # Gaussian Process Regressor Specific
    model_name = 'Gaussian Process Regressor Specific'
    model = GaussianProcessRegressor()
    X, y = extractData()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    bestModel(model, model_name, GPR_parameters, X_train, X_test, y_train, y_test)

    # ---------------------------------------------------------
    # Gaussian Process Regressor Top
    model_name = 'Gaussian Process Regressor Top 10'
    model = GaussianProcessRegressor()
    X, y = extractAllData()
    X = X[feature_imp[:10].index]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    bestModel(model, model_name, GPR_parameters, X_train, X_test, y_train, y_test)

    # ---------------------------------------------------------
    # Gradient Boosting Regressor All
    model_name = 'Gradient Boosting Regressor All'
    model = GradientBoostingRegressor()
    X, y = extractAllData()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    bestModel(model, model_name, GBR_parameters, X_train, X_test, np.ravel(y_train), np.ravel(y_test))



main()
