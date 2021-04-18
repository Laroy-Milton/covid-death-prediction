from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from utils import extractAllData, extractData, plot_learning_curve

def run_mlp(X, y):
    y = np.ravel(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    pipeline = Pipeline([('imputer', SimpleImputer()), ('scaler', StandardScaler())])
    X_train_scaled = pipeline.fit_transform(X_train)
    X_test_scaled = pipeline.transform(X_test)

    # Optimal parameter testing
    mlp = MLPRegressor(max_iter=20000)
    parameter_space = {
        'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant', 'adaptive'],
    }
    params_Ridge = {'alpha': [1, 0.1, 0.01, 0.001, 0.0001, 0], "fit_intercept": [True, False],
                    "solver": ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}
    gridsearch = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
    gridsearch.fit(X_train_scaled, y_train)
    clf = gridsearch.best_estimator_
    print('Best hyperparameters found:\n', gridsearch.best_params_)

    # Not the best representation of the results, but good for comparing
    score = clf.score(X_test_scaled, y_test)
    print("Score: ", score)
    y_pred = clf.predict(X_test_scaled)
    rmse = mean_squared_error(y_test, y_pred)
    print("RMSE:", np.sqrt(rmse))
    print()

    # Learning curve plot
    cv = ShuffleSplit(n_splits=100, test_size=0.3)
    plot_learning_curve(clf, "MLP", X_train_scaled, y_train, cv=cv, n_jobs=4)
    plt.show()

print("Multi-layer Perceptron Regressor All")
X,y = extractAllData()
run_mlp(X, y)

print("Multi-layer Perceptron Regressor Specific")
X,y = extractData()
run_mlp(X, y)
