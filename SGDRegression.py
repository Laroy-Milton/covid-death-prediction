import pandas as pd
import numpy as np
from functools import reduce
from sklearn import linear_model, preprocessing
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, ShuffleSplit, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None, n_jobs=None,
                        train_sizes=np.linspace(.1, 1.0, 5)):
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
    y = merged[['W60_new_deaths_per_million']]
    cols = [c for c in merged.columns if c.lower()[:4] != 'coun']
    X = merged[cols]
    X = X.drop(['iso_code', 'W60_new_deaths_per_million'], axis=1)
    return X, y


def main():
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
    dataframe_list = [demographics, economics, fitness, health, tourism]
    merged = reduce(lambda left, right: pd.merge(left, right, on='Country Code'), dataframe_list)
    merged = pd.merge(merged, covid, left_on='Country Code', right_on='iso_code')

    y = merged[['W60_new_deaths_per_million']]
    X = merged.drop(['Country Code', 'iso_code', 'W60_new_deaths_per_million'], axis=1)
    # X, y = extractAllData()
    y = np.ravel(y)
    # X = preprocessing.normalize(X, norm='l2')
    pd.set_option('display.max_rows', X.shape[0] + 1)
    # X = scale(X)
    # y = scale(y)
    scalar = MinMaxScaler()
    X = scalar.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    X_train = preprocessing.normalize(X_train, norm='l2')

    # parameter_space = {
    #     'loss': ["squared_loss", "huber", "epsilon_insensitive"],
    #     'penalty': ["none", "l2", "l1"],
    #     'shuffle': [True, False],
    #     "average": [True, False],
    # }
    # clf = GridSearchCV(SGDR, parameter_space)
    # clf.fit(X_train, y_train)
    # print('Best parameters found:\n', clf.best_params_)

    # SGD
    SGD = linear_model.SGDRegressor(loss="squared_loss", penalty="l2", average=False, shuffle=True)
    SGD.fit(X_train, y_train)
    SGDscore = SGD.score(X_test, y_test)
    print("SGD test score: %.2f%%" % (SGDscore * 100))

    # SVR
    svr = SVR()
    svr.fit(X_train, y_train)
    SVRscore = svr.score(X_test, y_test)
    print("SVR test score: %.2f%%" % (SVRscore * 100))

    # KNN
    knn = KNeighborsRegressor()
    knn.fit(X_train, y_train)
    KNNscore = knn.score(X_test, y_test)
    print("KNN test score: %.2f%%" % (KNNscore * 100))

    # KRR
    krr = KernelRidge()
    krr.fit(X_train, y_train)
    KRRscore = krr.score(X_test, y_test)
    print("KRR test score: %.2f%%" % (KRRscore * 100))

    cv = ShuffleSplit(n_splits=100, test_size=0.3)
    plot_learning_curve(krr, "title", X, y, cv=cv, n_jobs=4)
    plt.show()

    # x_ax = range(len(y_test))
    # plt.plot(x_ax, y_test, linewidth=1, label="original")
    # plt.plot(x_ax, ypred, linewidth=1.1, label="predicted")
    # plt.title("y-test and y-predicted data")
    # plt.xlabel('X-axis')
    # plt.ylabel('Y-axis')
    # plt.legend(loc='best', fancybox=True, shadow=True)
    # plt.grid(True)
    # plt.show()


main()
