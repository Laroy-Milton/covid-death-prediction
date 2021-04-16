# TODO Delete this file later, just keeping it for reference for the time being

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

from functools import reduce
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, ShuffleSplit, train_test_split
from sklearn import metrics
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import Normalizer, MinMaxScaler


def plot_learning_curves(model, X_train, y_train, X_val, y_val, title):
  train_errors, val_errors = [], []
  for m in range(4, len(X_train)):
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


def best_random_search(XTrain, yTrain):
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]

    max_features = ['auto', 'sqrt', 'log2']

    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)

    min_samples_split = [2, 5, 10]

    min_samples_leaf = [1, 2, 4]

    bootstrap = [True, False]

    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    rnd_clf = RandomForestRegressor()
    rf_random = RandomizedSearchCV(estimator=rnd_clf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                                   random_state=42, n_jobs=-1)
    rf_random.fit(XTrain, np.ravel(yTrain))
    print('\nBest parameters:', rf_random.best_params_)
    return rf_random

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

    return (X, y)



def extractAllData():
    # Load all csv files into pandas
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

    #TODO this gets rid of any values that have 0 in the y column
    merged = merged.loc[~(merged['W60_new_deaths_per_million'] == 0)]

    y = merged[['W60_new_deaths_per_million']]
    cols = [c for c in merged.columns if c.lower()[:4] != 'coun']
    X = merged[cols]
    X = X.drop(['iso_code', 'W60_new_deaths_per_million'], axis=1)

    X.to_csv("X.csv")
    y.to_csv("y.csv")
    return X, y

def main():

    X, y = extractAllData()



    # for col_name in X.columns:
    #     print(col_name)

    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.3)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(XTrain)

    RFR = RandomForestRegressor(n_estimators=50)

    scaler.transform(XTest)
    RFR.fit(scaled, np.ravel(yTrain))

    prediction = RFR.predict(XTest)

    print(RFR.score(XTest, prediction))
    print(prediction)


    # cv = ShuffleSplit(n_splits=100, test_size=0.3)
    # plot_learning_curve(RFR, "title", X, y,  # axes=axes[:, 1], ylim=(0.7, 1.01),
    #                     cv=cv, n_jobs=4)
    # plt.show()


    # feature_imp =pd.Series(RFR.feature_importances_, index=list(X)).sort_values(ascending=False)
    # sn.barplot(x=feature_imp[:10], y=feature_imp[:10].index)
    # plt.title('Top 10 important features')
    # plt.tight_layout()
    # # print("Top 10 features")
    # # print(feature_imp[:10])
    #
    # plt.show()
    #
    # sn.barplot(x=feature_imp, y=feature_imp.index)
    # plt.title('Important features ranked')
    # plt.tight_layout()
    # plt.show()
    # # print("All features")
    # # print(feature_imp)

    # RFR = RandomForestRegressor(n_estimators=50)
    # RFR.fit(XTrain, np.ravel(yTrain))
    #
    # Xnew = X.iloc[0]
    # Xnew = Xnew.to_numpy()
    # prediction = RFR.predict(Xnew.reshape(1,-1))
    # print(prediction)
    # print(y.iloc[0])
    # print(RFR.score(XTest, prediction))
    # plot_learning_curves(RFR, XTrain, yTrain, XTest, yTest, 'Random Forest Regression')
    # plt.show()





main()


