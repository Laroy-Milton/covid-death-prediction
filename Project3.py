from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from utils import *

# These are defined in utils.py
# SAVE = True   # When SAVE = True, plot's are saved instead of shown
# seed = 1972
# n_jobs = 2    # if set to -1 it will use all processors

def main():
    with open("Output.txt", 'w') as file:
        RFR_parameters = {'n_estimators': [25, 40, 50, 60, 70, 80],
                          'criterion': ['mae'],
                          'max_depth': [25, 75, 80, 100, None],
                          'min_samples_split': np.arange(2, 5),
                          'min_samples_leaf': [6, 7, 9],
                          'max_features': ['auto', 'sqrt', 'log2'],
                          'bootstrap': [True, False],
                          'n_jobs': [n_jobs],
                          'random_state': [seed]}

        SVR_parameters = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                          'degree': [2, 3, 4, 5],
                          'gamma': ['scale', 'auto'],
                          'epsilon': [0.005, 0.01, 0.25, 0.5, 0.1, 1.0],
                          'shrinking': [True, False]}

        ridge_parameters = {'alpha': [0.1, 0.5, 1.0, 1.5, 3, 5, 7, 10.0, 12, 15],
                            'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
                            'max_iter': [800, 1000, 2500, 5000, 7500, 10000],
                            'random_state': [seed]}

        # Decision Tree Regression
        max_depth = [int(x) for x in np.arange(1, 10)]
        max_depth.append(None)
        DTR_parameters = {'splitter': ['best', 'random'],
                          'max_depth': max_depth,
                          'min_samples_split': np.arange(10, 40, 2),
                          'min_samples_leaf': np.arange(2, 20, 3),
                          'min_weight_fraction_leaf': [0, 0.5],
                          'max_features': ['auto', 'sqrt', 'log2'],
                          'random_state': [seed]}

        # Gaussian process Regressor
        GPR_parameters = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 20],
                          'n_restarts_optimizer': np.arange(0, 5),
                          'random_state': [seed]}

        # Gradient Boosting Regressor
        GBR_parameters = {'loss': ['ls', 'lad', 'huber', 'quantile'],
                          'learning_rate': [0.001, 0.01, 0.1, 1.0],
                          'n_estimators': [50, 100, 200],
                          'min_samples_split': [10, 20, 30],
                          'min_samples_leaf': [10, 15, 20],
                          'min_weight_fraction_leaf': [0, 0.5],
                          'max_depth': np.arange(1, 4),
                          'max_features': ['auto', 'sqrt', 'log2'],
                          'alpha': [0.001, 0.01, 0.1, 0.99]}


        # ---------------------------------------------------------
        # Data Vis
        X_all, y = extractAllData(save=True)
        y = np.ravel(y)
        plotData(X_all)

        X_spec = extractDataSpec(X_all)

        # ---------------------------------------------------------
        # Random Forest Regression Specific
        model_name = 'Random Forest Regression All'
        model = RandomForestRegressor()

        X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=0.30, random_state=seed)

        bm = bestModel(model, model_name, RFR_parameters, X_train, X_test, y_train, y_test, file, seed)
        feature_imp = topFeatures(bm, (X_train, y_train), file, num_features=10)

        X_feat = X_all[feature_imp[:10].index]

        # ---------------------------------------------------------
        # Random Forest Regression Specific
        model_name = 'Random Forest Regression specific'
        model = RandomForestRegressor()

        X_train, X_test, y_train, y_test = train_test_split(X_spec, y, test_size=0.30, random_state=seed)
        bestModel(model, model_name, RFR_parameters, X_train, X_test, y_train, y_test, file, seed)

        # ---------------------------------------------------------
        # Random Forest Regression Top
        model_name = 'Random Forest Regression Top'
        model = RandomForestRegressor()

        X_train, X_test, y_train, y_test = train_test_split(X_feat, y, test_size=0.30, random_state=seed)
        bestModel(model, model_name, RFR_parameters, X_train, X_test, y_train, y_test, file, seed)

        # ---------------------------------------------------------
        # SVR ALL
        model_name = 'SVR All'
        model = SVR()
        
        X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=0.30, random_state=seed)
        bestModel(model, model_name, SVR_parameters, X_train, X_test, y_train, y_test, file, seed)

        # ---------------------------------------------------------
        # SVR Specific
        model_name = 'SVR Specific'
        model = SVR()

        X_train, X_test, y_train, y_test = train_test_split(X_spec, y, test_size=0.30, random_state=seed)
        bestModel(model, model_name, SVR_parameters, X_train, X_test, y_train, y_test, file, seed)

        # ---------------------------------------------------------
        # SVR ALL
        model_name = 'SVR Top'
        model = SVR()

        X_train, X_test, y_train, y_test = train_test_split(X_feat, y, test_size=0.30, random_state=seed)
        bestModel(model, model_name, SVR_parameters, X_train, X_test, y_train, y_test, file, seed)


        file.flush()
        # ---------------------------------------------------------
        # Ridge Regression ALL
        model_name = 'Ridge Regression ALL'
        model = Ridge()

        X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=0.30, random_state=seed)
        bestModel(model, model_name, ridge_parameters, X_train, X_test, y_train, y_test, file, seed)

        # ---------------------------------------------------------
        # Ridge regression Specific
        model_name = 'Ridge regression Specific'
        model = Ridge()

        X_train, X_test, y_train, y_test = train_test_split(X_spec, y, test_size=0.30, random_state=seed)
        bestModel(model, model_name, ridge_parameters, X_train, X_test, y_train, y_test, file, seed)

        # ---------------------------------------------------------
        # Ridge regression Top 10 Feature importance
        model_name = 'Ridge Regression Top 10'
        model = Ridge()

        X_train, X_test, y_train, y_test = train_test_split(X_feat, y, test_size=0.30, random_state=seed)
        bestModel(model, model_name, ridge_parameters, X_train, X_test, y_train, y_test, file, seed)

        file.flush()
        # ---------------------------------------------------------
        # Decision tree Regression All
        model_name = 'Decision Tree Regressor All'
        model = DecisionTreeRegressor()

        X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=0.30, random_state=seed)
        bestModel(model, model_name, DTR_parameters, X_train, X_test, y_train, y_test, file, seed)

        # ---------------------------------------------------------
        # Decision tree Regression Specific
        model_name = 'Decision Tree Regressor Specific'
        model = DecisionTreeRegressor()

        X_train, X_test, y_train, y_test = train_test_split(X_spec, y, test_size=0.30, random_state=seed)
        bestModel(model, model_name, DTR_parameters, X_train, X_test, y_train, y_test, file, seed)

        # ---------------------------------------------------------
        # Decision tree Regression Top 10 Feature importance
        model_name = 'Decision Tree Regressor Top 10'
        model = DecisionTreeRegressor()

        X_train, X_test, y_train, y_test = train_test_split(X_feat, y, test_size=0.30, random_state=seed)
        bestModel(model, model_name, DTR_parameters, X_train, X_test, y_train, y_test, file, seed)

        file.flush()
        # ---------------------------------------------------------
        # Gaussian Process Regressor All
        model_name = 'Gaussian Process Regressor All'
        model = GaussianProcessRegressor()

        X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=0.30, random_state=seed)
        bestModel(model, model_name, GPR_parameters, X_train, X_test, y_train, y_test, file, seed)

        # ---------------------------------------------------------
        # Gaussian Process Regressor Specific
        model_name = 'Gaussian Process Regressor Specific'
        model = GaussianProcessRegressor()

        X_train, X_test, y_train, y_test = train_test_split(X_spec, y, test_size=0.30, random_state=seed)
        bestModel(model, model_name, GPR_parameters, X_train, X_test, y_train, y_test, file, seed)

        # ---------------------------------------------------------
        # Gaussian Process Regressor Top
        model_name = 'Gaussian Process Regressor Top 10'
        model = GaussianProcessRegressor()

        X_train, X_test, y_train, y_test = train_test_split(X_feat, y, test_size=0.30, random_state=seed)
        bestModel(model, model_name, GPR_parameters, X_train, X_test, y_train, y_test, file, seed)


        file.flush()
        # ---------------------------------------------------------
        # Gradient Boosting Regressor All
        model_name = 'Gradient Boosting Regressor All'
        model = GradientBoostingRegressor()

        X_train, X_test, y_train, y_test = train_test_split(X_feat, y, test_size=0.30, random_state=seed)
        bestModel(model, model_name, GBR_parameters, X_train, X_test, y_train, y_test, file, seed)
        
        # ---------------------------------------------------------
        # Gradient Boosting Regressor Specific
        model_name = 'Gradient Boosting Regressor Specific'
        model = GradientBoostingRegressor()

        X_train, X_test, y_train, y_test = train_test_split(X_spec, y, test_size=0.30, random_state=seed)
        bestModel(model, model_name, GBR_parameters, X_train, X_test, y_train, y_test, file, seed)

        # ---------------------------------------------------------
        # Gradient Boosting Regressor Top
        model_name = 'Gradient Boosting Regressor Specific'
        model = GradientBoostingRegressor()

        X_train, X_test, y_train, y_test = train_test_split(X_feat, y, test_size=0.30, random_state=seed)
        bestModel(model, model_name, GBR_parameters, X_train, X_test, y_train, y_test, file, seed)


main()
