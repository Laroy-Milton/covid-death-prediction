import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

from functools import reduce

from sklearn.model_selection import learning_curve, RepeatedKFold, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, QuantileTransformer, StandardScaler
from sklearn.metrics import make_scorer, mean_squared_error, r2_score

# When SAVE = True, plot's are saved instead of shown
SAVE = False
seed = 1972

# if set to -1 it will use all processors
n_jobs = 2

np.random.seed(seed)


# Source https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
def plot_learning_curve(estimator, title, X, y, seed, cv=None, train_sizes=np.linspace(.1, 1.0, 5)):
    _, axes = plt.subplots(1, 1)

    axes.set_title(title)

    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, return_times=True, random_state=seed)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot learning curve
    axes.grid()
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                      train_scores_mean + train_scores_std, alpha=0.1,
                      color="r")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                      test_scores_mean + test_scores_std, alpha=0.1,
                      color="g")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
              label="Training score")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g",
              label="Cross-validation score")
    axes.legend(loc="best")

    return plt


def plotData(X):
    X_trans = {'Raw Data': X,
               'Standard Scaler': StandardScaler().fit_transform(X),
               'Min Max': MinMaxScaler().fit_transform(X),
               'yeo-johnson': PowerTransformer(method='yeo-johnson', standardize=True).fit_transform(X),
               'Quantile normal': QuantileTransformer(n_quantiles=75, output_distribution='normal').fit_transform(X),
               'Quantile uniform': QuantileTransformer(n_quantiles=75, output_distribution='uniform').fit_transform(X)}

    for name, x in X_trans.items():
        plt.hist(x)
        plt.title(name)

        save_name = name.replace(' ', '_').strip()
        plt.savefig(save_name + ".png") if SAVE else plt.show()


def extractDataSpec(X):
    return X[['Population density (people per sq. km of land area) ', 'Population, total',
              'Current health expenditure (% of GDP)',
              'Life expectancy at birth',
              'Physicians (per 1,000 people)', 'Nurses and midwives (per 1,000 people)',
              'International tourism, number of arrivals', 'Net migration']]


# If save True then csv for x and y are saved to file
def extractAllData(save=False):
    # Load all data from csv files into pandas
    covid = pd.read_csv('DataFiles\Covid-60weeks.csv')[['iso_code', 'W60_new_deaths_per_million']]
    demographics = pd.read_csv('DataFiles\Demographics.csv')
    economics = pd.read_csv('DataFiles\Economics.csv')
    fitness = pd.read_csv('DataFiles\Fitness.csv')
    health = pd.read_csv('DataFiles\Health.csv')
    sanitation = pd.read_csv('DataFiles\Sanitation.csv')
    tourism = pd.read_csv('DataFiles\Tourism.csv')

    dataframe_list = [demographics, economics, fitness, health, tourism]
    merged = reduce(lambda left, right: pd.merge(left, right, on='Country Code'), dataframe_list)
    merged = pd.merge(merged, covid, left_on='Country Code', right_on='iso_code')

    # this gets rid of any values that have 0 and replaces it with the column mean
    merged.replace(0, merged.median(axis=0), inplace=True)

    y = merged[['W60_new_deaths_per_million']]
    cols = [c for c in merged.columns if c.lower()[:4] != 'coun']
    X = merged[cols]
    X = X.drop(['iso_code', 'W60_new_deaths_per_million'], axis=1)

    X.to_csv("X.csv") if save else None
    y.to_csv("y.csv") if save else None
    return X, y


# Uses GridhSearch to find the best parameters then plots the learning curve
def bestModel(model, model_name, params, X_train, X_test, y_train, y_test, file, seed):
    print('\nTraining ' + model_name + '...')
    file.write('\nTraining ' + model_name + '...')

    RMSE = make_scorer(mean_squared_error, greater_is_better=False)

    # qt = QuantileTransformer(n_quantiles=75, output_distribution='normal')
    # X_train = qt.fit_transform(X_train)
    # X_test = qt.transform(X_test)

    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    minMax = MinMaxScaler()
    X_train = minMax.fit_transform(X_train)
    X_test = minMax.transform(X_test)

    cv = RepeatedKFold(n_splits=10, n_repeats=100)

    gs = RandomizedSearchCV(model, params, n_iter=75, cv=cv, n_jobs=n_jobs, return_train_score=True)
    gs.fit(X_train, y_train)

    trainScore = gs.best_score_
    testScore = gs.score(X_test, y_test)

    title = '{} Train: {:.2} Test: {:.2}'.format(model_name, trainScore, testScore)
    print(title)
    print(gs.best_params_)
    file.write("\n" + title)
    file.write("\n" + str(gs.best_params_))

    plot = plot_learning_curve(gs.best_estimator_, title, X_train, y_train, cv=cv, seed=seed)

    save_name = model_name.replace(' ', '_').strip()
    plot.savefig(save_name + ".png") if SAVE else plot.show()

    pred = gs.predict(X_test)

    print("TRAIN", trainScore)
    print('TEST', testScore)
    file.write("\nTRAIN " + str(trainScore))
    file.write('\nTEST ' + str(testScore))

    print('RMSE score on test:', np.sqrt(mean_squared_error(y_test, pred)))
    print('r2 score on test:', r2_score(y_test, pred))
    file.write('\nRMSE score on test: ' + str(np.sqrt(mean_squared_error(y_test, pred))))
    file.write('\nr2 score on test: ' + str(r2_score(y_test, pred)) + '\n')

    return gs


# Finds the top features when given results from gridSearch on random forest and Plots the top features
def topFeatures(model, data, file, num_features=10):
    X, y = data
    title = 'Top ' + str(num_features) + ' Features'
    feature_imp = pd.Series(model.best_estimator_.feature_importances_, index=list(X)).sort_values(ascending=False)
    sn.barplot(x=feature_imp[:num_features], y=feature_imp[:num_features].index)
    plt.title(title)
    plt.tight_layout()
    print()
    print(title)
    print(feature_imp[:10])
    print()

    file.write("\n\n" + title + "\n")
    file.writelines(str(feature_imp[:10]))
    plt.savefig("Feature_importance.png") if SAVE else plt.show()

    return feature_imp
