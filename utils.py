import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

from functools import reduce

from sklearn import metrics
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV


# TODO throw away
def plotLearningCurve(gs, title):
    test_scores = gs.cv_results_['mean_test_score']
    train_scores = gs.cv_results_['mean_train_score']

    plt.plot(test_scores, label='test')
    plt.plot(train_scores, label='train')
    plt.legend(loc='best')
    plt.title(title)
    plt.show()


# TODO throw away
def plot_learning_curves(model, X_train, y_train, X_val, y_val, title):
    train_errors, val_errors = [], []
    for m in range(5, len(X_train)):
        model.fit(X_train[:m], np.ravel(y_train[:m]))
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(metrics.mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(metrics.mean_squared_error(y_val, y_val_predict))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label='train')
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label='val')
    plt.xlabel('Training set size')
    plt.ylabel('RMSE')
    plt.legend()
    plt.title(title)


# Source https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
  Generate 3 plots: the test and training learning curve, the training
  samples vs fit times curve, the fit times vs score curve.

  Parameters
  ----------
  estimator : estimator instance
      An estimator instance implementing `fit` and `predict` methods which
      will be cloned for each validation.

  title : str
      Title for the chart.

  X : array-like of shape (n_samples, n_features)
      Training vector, where ``n_samples`` is the number of samples and
      ``n_features`` is the number of features.

  y : array-like of shape (n_samples) or (n_samples, n_features)
      Target relative to ``X`` for classification or regression;
      None for unsupervised learning.

  axes : array-like of shape (3,), default=None
      Axes to use for plotting the curves.

  ylim : tuple of shape (2,), default=None
      Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

  cv : int, cross-validation generator or an iterable, default=None
      Determines the cross-validation splitting strategy.
      Possible inputs for cv are:

        - None, to use the default 5-fold cross-validation,
        - integer, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

      For integer/None inputs, if ``y`` is binary or multiclass,
      :class:`StratifiedKFold` used. If the estimator is not a classifier
      or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

      Refer :ref:`User Guide <cross_validation>` for the various
      cross-validators that can be used here.

  n_jobs : int or None, default=None
      Number of jobs to run in parallel.
      ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
      ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
      for more details.

  train_sizes : array-like of shape (n_ticks,)
      Relative or absolute numbers of training examples that will be used to
      generate the learning curve. If the ``dtype`` is float, it is regarded
      as a fraction of the maximum size of the training set (that is
      determined by the selected validation method), i.e. it has to be within
      (0, 1]. Otherwise it is interpreted as absolute sizes of the training
      sets. Note that for classification the number of samples usually have
      to be big enough to contain at least one sample from each class.
      (default: np.linspace(0.1, 1.0, 5))
  """
    y = np.ravel(y)
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt


def extractData():
    # Load all csv files into pandas
    covid = pd.read_csv('DataFiles\Covid-60weeks.csv')[['iso_code', 'W60_new_deaths_per_million']]
    demographics = pd.read_csv('DataFiles\Demographics.csv')[
        ['Country Code', 'Population density (people per sq. km of land area) ', 'Population, total']]
    economics = pd.read_csv('DataFiles\Economics.csv')[['Country Code', 'Current health expenditure (% of GDP)']]
    fitness = pd.read_csv('DataFiles\Fitness.csv')[['Country Code', 'Life expectancy at birth']]
    health = pd.read_csv('DataFiles\Health.csv')[
        ['Country Code', 'Physicians (per 1,000 people)', 'Nurses and midwives (per 1,000 people)']]
    sanitation = pd.read_csv('DataFiles\Sanitation.csv')
    tourism = pd.read_csv('DataFiles\Tourism.csv')[
        ['Country Code', 'International tourism, number of arrivals', 'Net migration']]

    # Merge everything needed except covid for now because it is the only table with a different column name
    # The biggest drop in the amount of data comes from sanitation, and only a slight drop is found in tourism.
    # The rest of the datasets remove around 2 to 3 countries.,
    dataframe_list = [demographics, economics, fitness, health, tourism]
    merged = reduce(lambda left, right: pd.merge(left, right, on='Country Code'), dataframe_list)
    merged = pd.merge(merged, covid, left_on='Country Code', right_on='iso_code')
    # print(merged)

    y = merged[['W60_new_deaths_per_million']]
    X = merged.drop(['Country Code', 'iso_code', 'W60_new_deaths_per_million'], axis=1)

    return X, y


def extractAllData():
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

    # TODO this gets rid of any values that have 0 in the y column
    merged = merged.loc[~(merged['W60_new_deaths_per_million'] == 0)]

    y = merged[['W60_new_deaths_per_million']]
    cols = [c for c in merged.columns if c.lower()[:4] != 'coun']
    X = merged[cols]
    X = X.drop(['iso_code', 'W60_new_deaths_per_million'], axis=1)

    X.to_csv("X.csv")
    y.to_csv("y.csv")
    return X, y


# Uses GridhSearch to find the best parameters then plots the learning curve
def bestModel(model, model_name, params, X, y, cv=None):
    print('Training ' + model_name + '...')

    gs = GridSearchCV(model, params, cv=cv, scoring='r2', return_train_score=True)
    gs.fit(X, y)

    title = '{} {:.2}'.format(model_name, gs.best_score_)
    print(title)
    plot_learning_curve(gs.best_estimator_, title, X, y)

    return gs


# Finds the top features when given results from gridSearch on random forest and Plots the top features
def topFeatures(model, data, num_features=10):
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
    plt.show()
    return feature_imp
