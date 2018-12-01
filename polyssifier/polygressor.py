#! /usr/bin/env python
from sklearn.datasets import make_regression, load_diabetes
import sys
import numpy as np
import pickle as p
from multiprocessing import Manager, Pool
import os
import pandas as pd
from copy import deepcopy
from sklearn.model_selection import StratifiedKFold, GridSearchCV, KFold
from sklearn.metrics import (f1_score, confusion_matrix, roc_auc_score,
                             mean_squared_error, r2_score)
from sklearn.externals import joblib
import time
from sklearn.preprocessing import LabelEncoder
from itertools import starmap
from .poly_utils import (build_classifiers, MyVoter, build_regressors,
                        MyRegressionMedianer)
from .report import Report
from .logger import make_logger
from .default_include import DEFAULT_include
from .polysis import Polysis

sys.setrecursionlimit(10000)
logger = make_logger('polygressor')

PERMITTED_SCORINGS = ['r2', 'mse']
DEFAULT_do_regress = False
DEFAULT_n_folds = 2
DEFAULT_scale = True
DEFAULT_feature_selection = False
DEFAULT_save = True
DEFAULT_scoring = 'r2'
DEFAULT_project_name = ''
DEFAULT_concurrency = 1
DEFAULT_verbose = True
DEFAULT_num_degrees = 1
DEFAULT_path = '.'


class Polygressor(Polysis):
    """Polyssifier - wrapper for sklearn classifiers and regressors.
       Run multiple models on data, and output a report.

       | variable     | description   | dtype       | expected vals     |
       | ------------ | ------------- | ----------- | ----------------- |
       | data         |   feature set | np.ndarray  | ----------------- |
       | label        |   label set   | np.ndarray  | ----------------- |
       | n_folds      | # of folds    | int         | any integer       |
       | scale        | scale data?   | boolean     | any boolean       |
       | scoring      | score metric  | str         | 'auc', ...        |
       | project_name | folder name   | str         | any slugified str |
       | concurrency  | # workers     | int         | any int below max |
       | verbose      | verbose out   | boolean     | any boolean       |
       | num_degrees  |degree of poly | int         | any int           |
    """

    def __init__(self, data, label, do_regress=DEFAULT_do_regress,
                 n_folds=DEFAULT_n_folds, scale=DEFAULT_scale,
                 include=DEFAULT_include,
                 feature_selection=DEFAULT_feature_selection,
                 save=DEFAULT_save, scoring=DEFAULT_scoring,
                 project_name=DEFAULT_project_name,
                 concurrency=DEFAULT_concurrency, verbose=DEFAULT_verbose,
                 num_degrees=DEFAULT_num_degrees, path=DEFAULT_path):
        super(Polygressor, self).__init__(data, label, n_folds=n_folds, scale=scale,
                                          include=include, feature_selection=feature_selection, save=save, scoring=scoring,
                                          project_name=project_name, concurrency=concurrency, verbose=verbose,
                                          num_degrees=num_degrees, path=path)
        if self.scoring not in PERMITTED_SCORINGS:
            self.scoring = PERMITTED_SCORINGS[0]

    def initialize_folds(self):
        """Overrides abstract method"""
        self.stratified_k_fold = KFold(n_splits=self.n_folds,
                                       random_state=1988)
        self.finalize_folds()

    def initialize_models(self):
        """Overrides abstract method"""
        logger.info('Building regressors ...')
        self.models = build_regressors(self.include,
                                       self.scale,
                                       self.feature_selection,
                                       self.data.shape[1])

    def gather_results(self):
        '''Gathers results'''
        for clf_name in self.models:
            self.coefficients[clf_name] = []
            temp = np.zeros((self.n_class, self.n_class))
            temp_pred = np.zeros((self.data.shape[0], ))
            temp_prob = np.zeros((self.data.shape[0], ))
            clfs = self.fitted_models[clf_name]
            for n in range(self.n_folds):
                train_score, test_score, prediction, prob,\
                    coefs, fitted_model = self.result.pop(0)
                clfs.append(fitted_model)
                self.scores.loc[n, (clf_name, 'train')] = train_score
                self.scores.loc[n, (clf_name, 'test')] = test_score
                temp_prob[self.k_fold[n][1]] = prob
                temp_pred[self.k_fold[n][1]] = prediction
                self.coefficients[clf_name].append(coefs)
            self.predictions[clf_name] = temp_pred
            self.test_prob[clf_name] = temp_prob

    def _scorer(self, clf, X, y):
        '''Function that scores a regressor according to what is available as a
        predict function.
        Input:
            - reg = Fitted regressor object
            - X = input data matrix
            - y = corresponding values to the data matrix
        Output:
            - The mean sqaure error or r squared value for the given regressor and data. The default scoring is
            r squared value.
        '''
        if self.scoring == 'mse':
            return mean_squared_error(y, clf.predict(X))
        else:
            return r2_score(y, clf.predict(X))

    def aggregate_results(self):
        """Overrides abstract method"""
        self.compute_median()

    def feature_importance(self, clf):
        # Feature importance
        if hasattr(clf, 'steps'):
            temp = clf.steps[-1][1]
        elif hasattr(clf, 'best_estimator_'):
            if hasattr(clf.best_estimator_, 'steps'):
                temp = clf.best_estimator_.steps[-1][1]
            else:
                temp = clf.best_estimator_
        try:
            if hasattr(temp, 'coef_'):
                coefficients = temp.coef_
            elif hasattr(temp, 'feature_importances_'):
                coefficients = temp.feature_importances_
            else:
                coefficients = None
        except:
            coefficients = None
        return coefficients

    def compute_median(self):
        self.fitted_models = pd.DataFrame(self.fitted_models)
        self.scores['Median', 'train'] = np.zeros((self.n_folds, ))
        self.scores['Median', 'test'] = np.zeros((self.n_folds, ))
        temp_pred = np.zeros((self.data.shape[0], ))
        for n, (train, test) in enumerate(self.k_fold):
            reg = MyRegressionMedianer(self.fitted_models.loc[n].values)
            X, y = self.data[train, :], self.label[train]
            self.scores.loc[n, ('Median', 'train')] = self._scorer(reg, X,
                                                                   y)
            X, y = self.data[test, :], self.label[test]
            self.scores.loc[n, ('Median', 'test')] = self._scorer(reg, X,
                                                                  y)
            temp_pred[test] = reg.predict(X)
        self.predictions['Median'] = temp_pred


if __name__ == "__main__":
    NSAMPLES = 10000
    data, label = load_diabetes(return_X_y=True)
    poly = Polygressor(data, label, n_folds=5, verbose=1,
                       feature_selection=False,
                       save=False, project_name='polygressor_runtest')
    poly.build()
    poly.run()
