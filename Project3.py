import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

from functools import reduce
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV, ShuffleSplit, train_test_split
from sklearn import metrics
from sklearn.model_selection import learning_curve
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import Normalizer, MinMaxScaler



from utils import *




def main():
    RFR_parameters = {'n_estimators': np.arange(2, 5),
                     'criterion': ['mse', 'mae'],
                     'min_samples_split': np.linspace(0.1, 1, num=10),
                     'max_features': ['auto', 'sqrt', 'log2'],
                     'bootstrap': [True, False],
                     'n_jobs': [-1]}

    ridge_parameters = {'alpha': np.linspace(0.0, 1.0, num=30),
                        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
                        'max_iter': [500, 1000, 10000]}

    KNN_parameters = {'n_neighbors': [30],  # np.arange(2,10),
                      'weights': ['uniform', 'distance'],
                      'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                      'leaf_size': np.arange(2, 30),
                      'p': [1, 2],
                      'n_jobs': [-1]}



    X, y = extractAllData()

    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.3)


    RFR = RandomForestRegressor()
    gs = GridSearchCV(RFR, RFR_parameters, return_train_score=True)
    gs.fit(XTrain, np.ravel(yTrain))
    #plotLearningCurve(gs, 'Random Forest Regression')


    feature_imp =pd.Series(gs.best_estimator_.feature_importances_, index=list(XTrain)).sort_values(ascending=False)
    sn.barplot(x=feature_imp[:10], y=feature_imp[:10].index)
    plt.title('Top 10 important features')
    plt.tight_layout()
    # print("Top 10 features")
    # print(feature_imp[:10])

    plt.show()


    print("Random Forest ALL score")
    print(gs.score(XTest, np.ravel(yTest)))


    #not all
    X, y = extractData()

    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.3)


    reg = Ridge(normalize=True)

    gs = GridSearchCV(reg, ridge_parameters, return_train_score=True)
    gs.fit(XTrain, yTrain)
    # p = gs.predict(XTest)
    plotLearningCurve(gs, 'Ridge regression Specifc')


    print("Ridge handpicked score")
    print(gs.score(XTest, yTest))



    # all data
    X, y = extractAllData()

    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.3)


    reg = Ridge()

    gs = GridSearchCV(reg, ridge_parameters, return_train_score=True)
    gs.fit(XTrain, yTrain)

    plotLearningCurve(gs, 'Ridge regression ALL')

    print("Ridge Regression ALL score")
    print(gs.score(XTest, yTest))




    # with new features
    X, y = extractAllData()
    X = X[feature_imp[:10].index]

    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.3)

    reg = Ridge()

    gs = GridSearchCV(reg, ridge_parameters, return_train_score=True)
    gs.fit(XTrain, yTrain)
    # p = gs.predict(XTest)
    plotLearningCurve(gs, 'Ridge regression Top 10 Feature importance')

    print("Ridge Regression Top 10 Important Features score")
    print(gs.score(XTest, yTest))



    #KNN
    X, y = extractAllData()

    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.3, random_state=10)


    knn = KNeighborsRegressor()

    gs = GridSearchCV(knn, KNN_parameters, return_train_score=True)
    gs.fit(XTrain, yTrain)
    # p = gs.predict(XTest)
    plotLearningCurve(gs, 'KNN Feature All')

    pred = gs.predict(XTest)
    print("KNN All: score")
    print(gs.score(XTest, yTest))




main()


