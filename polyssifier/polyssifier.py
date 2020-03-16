#! /usr/bin/env python
from sklearn.datasets import make_classification
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

PERMITTED_SCORINGS = ["auc"]
DEFAULT_do_regress = False
DEFAULT_n_folds = 2
DEFAULT_scale = True
DEFAULT_feature_selection = False
DEFAULT_save = True
DEFAULT_scoring = 'auc'
DEFAULT_project_name = 'polyssifier'
DEFAULT_concurrency = 1
DEFAULT_verbose = True
DEFAULT_num_degrees = 1
DEFAULT_path = '.'


class Polyssifier(Polysis):
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

    def __init__(self, data, label,
                 n_folds=DEFAULT_n_folds,
                 scale=DEFAULT_scale,
                 include=DEFAULT_include,
                 feature_selection=DEFAULT_feature_selection,
                 save=DEFAULT_save,
                 scoring=DEFAULT_scoring,
                 project_name=DEFAULT_project_name,
                 concurrency=DEFAULT_concurrency,
                 verbose=DEFAULT_verbose,
                 num_degrees=DEFAULT_num_degrees,
                 path=DEFAULT_path,
                 logger=None,
                 **kwargs):
        super(Polyssifier, self).__init__(data, label, n_folds=n_folds, scale=scale,
                                          include=include, feature_selection=feature_selection, save=save, scoring=scoring,
                                          project_name=project_name, concurrency=concurrency, verbose=verbose,
                                          num_degrees=num_degrees, path=path, logger=logger, **kwargs)
        if self.scoring not in PERMITTED_SCORINGS:
            self.scoring = PERMITTED_SCORINGS[0]

    def initialize_folds(self):
        """Overrides abstract method"""
        self.stratified_k_fold = StratifiedKFold(n_splits=self.n_folds,
                                                 random_state=1988,
                                                 shuffle=True)

    def initialize_models(self):
        """Overrides abstract method"""
        self.logger.info('Building classifiers ...')
        self._le = LabelEncoder()
        self._le.fit(self.label)
        self.label = self._le.transform(self.label)
        self.models = build_classifiers(self.include,
                                        self.scale,
                                        self.feature_selection,
                                        self.data.shape[1])

    def gather_results(self):
        """Overrides abstract method"""
        for clf_name in self.models:
            self.coefficients[clf_name] = []
            temp = np.zeros((self.n_class, self.n_class))
            temp_pred = np.zeros((self.data.shape[0], ))
            temp_prob = np.zeros((self.data.shape[0], ))
            clfs = self.fitted_models[clf_name]
            for n in range(self.n_folds):
                train_score, test_score, prediction, prob, confusion,\
                    coefs, fitted_model = self.result.pop(0)
                clfs.append(fitted_model)
                self.scores.loc[n, (clf_name, 'train')] = train_score
                self.scores.loc[n, (clf_name, 'test')] = test_score
                temp += confusion
                temp_prob[self.k_fold[n][1]] = prob
                temp_pred[self.k_fold[n][1]
                          ] = self._le.inverse_transform(prediction)
                self.coefficients[clf_name].append(coefs)
            self.confusions[clf_name] = temp
            self.predictions[clf_name] = temp_pred
            self.test_prob[clf_name] = temp_prob

    def feature_importance(self, clf):
        """Overrides abstract method"""
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

    def _scorer(self, clf, X, y):
        '''
        Overrides abstract method.

        Function that scores a classifier according to what is available as a
        predict function.
        Input:
            - clf = Fitted classifier object
            - X = input data matrix
            - y = estimated labels
        Output:
            - AUC score for binary classification or F1 for multiclass
        The order of priority is as follows:
            - predict_proba
            - decision_function
            - predict
        '''
        n_class = len(np.unique(y))
        if n_class == 2:
            if hasattr(clf, 'predict_proba'):
                ypred = clf.predict_proba(X)
                try:
                    ypred = ypred[:, 1]
                except:
                    self.logger.info(
                        'predict proba return shape{}'.format(ypred.shape))

                assert(len(ypred.shape) == 1)
            elif hasattr(clf, 'decision_function'):
                ypred = clf.decision_function(X)
                assert(len(ypred.shape) == 1)
            else:
                ypred = clf.predict(X)
            score = roc_auc_score(y, ypred)
        else:
            score = f1_score(y, clf.predict(X), average='weighted')
        return score

    def aggregate_results(self):
        """Overrides abstract method"""
        if len(self.include) > 1:
            self.perform_voting()

    def perform_voting(self):
        self.fitted_models = pd.DataFrame(self.fitted_models)
        self.scores['Voting', 'train'] = np.zeros((self.n_folds, ))
        self.scores['Voting', 'test'] = np.zeros((self.n_folds, ))
        temp = np.zeros((self.n_class, self.n_class))
        temp_pred = np.zeros((self.data.shape[0], ))
        for n, (train, test) in enumerate(self.k_fold):
            clf = MyVoter(self.fitted_models.loc[n].values)
            X, y = self.data[train, :], self.label[train]
            self.scores.loc[n, ('Voting', 'train')] = self._scorer(clf, X, y)
            X, y = self.data[test, :], self.label[test]
            self.scores.loc[n, ('Voting', 'test')] = self._scorer(clf, X, y)
            temp_pred[test] = clf.predict(X)
            temp += confusion_matrix(y, temp_pred[test])

        self.confusions['Voting'] = temp
        self.predictions['Voting'] = temp_pred
        self.test_prob['Voting'] = temp_pred

    def resolve_probabilities(self, X, clf, ypred):
        yprob = None
        if hasattr(clf, 'predict_proba'):
            # For compatibility with different sklearn versions
            yprob = clf.predict_proba(X)
            try:
                yprob = yprob[:, 1]
            except:
                self.logger.info(
                    'predict proba return shape {}'.format(yprob.shape))
        elif hasattr(clf, 'decision_function'):
            yprob = clf.decision_function(X)
            try:
                yprob = yprob[:, 1]
            except:
                self.logger.info(
                    'predict proba return shape {}'.format(yprob.shape))
            assert len(yprob.shape) == 1,\
                'predict proba return shape {}'.format(ypred.shape)
        return yprob

    def fit_model(self, args, clf_name, val, n_fold, project_name, save, scoring):
        train_score, test_score, ypred, yprob, coefficients, clf = super(
            Polyssifier, self).fit_model(args, clf_name, val, n_fold, project_name, save, scoring)
        train, test = args[0]['k_fold'][n_fold]
        X, y = self.get_xy(args, test)
        confusion = confusion_matrix(y, ypred)
        yprob = self.resolve_probabilities(X, clf, ypred)
        return (train_score, test_score,
                ypred, yprob,  # predictions and probabilities
                confusion,  # confusion matrix
                coefficients,  # Coefficients for feature ranking
                clf)  # fitted clf


if __name__ == "__main__":
    NSAMPLES = 100
    data, label = make_classification(n_samples=NSAMPLES, n_features=2,
                                      n_informative=1, n_redundant=1,
                                      n_repeated=0, n_classes=2,
                                      n_clusters_per_class=1, weights=None,
                                      flip_y=0.01, class_sep=1.0,
                                      hypercube=False, shift=0.0,
                                      scale=1.0, shuffle=True,
                                      random_state=1988)
    poly = Polyssifier(data, label, n_folds=5, verbose=1,
                       feature_selection=False,
                       save=False, project_name='polyssifier_runtest')
    poly.build()
    poly.run()
