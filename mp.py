from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from utils import *

def run_mlp(X, y, title):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=seed)
    pipeline = Pipeline([('scaler', StandardScaler())])
    X_train_scaled = pipeline.fit_transform(X_train)
    X_test_scaled = pipeline.transform(X_test)

    # Optimal parameter testing
    mlp = MLPRegressor(max_iter=20000)
    MLP_parameters = {
        'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant', 'adaptive'],
    }
    gridsearch = GridSearchCV(mlp, MLP_parameters, n_jobs=5, cv=3)
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
    plot = plot_learning_curve(gridsearch.best_estimator_, title, X_train, y_train, cv=cv, seed=seed)
    save_name = title.replace(' ', '_').strip()
    plot.savefig(PLOT_FOLDER + save_name + ".png") if SAVE else plot.show()
    plt.clf()
    plt.cla()
    plt.close()

# X_all, y = extractAllData(save=True)
# y = np.ravel(y)
# run_mlp(X_all, y)

X_all, y = extractAllData(save=True)
y = np.ravel(y)

X_spec = extractDataSpec(X_all)


MLP_parameters = {
        'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant', 'adaptive'],
    }


with open("mp_Output.txt", 'w') as file:
    # Multi-layer Perceptron Regressor All
    title = "Multi-layer Perceptron Regressor All"
    print(title)
    run_mlp(X_all, y, title)

    # Multi-layer Perceptron Regressor All
    title = "Multi-layer Perceptron Regressor Specific"
    print(title)
    run_mlp(X_spec, y, title)
