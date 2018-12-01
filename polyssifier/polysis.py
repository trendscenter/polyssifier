#! /usr/bin/env python
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
from poly_utils import (build_classifiers, MyVoter, build_regressors,
                        MyRegressionMedianer)
from default_include import DEFAULT_include
from report import Report
from logger import make_logger
from abc import ABC, abstractmethod
sys.setrecursionlimit(10000)
logger = make_logger('polyssifier')

PERMITTED_SCORINGS = []
DEFAULT_n_folds = 2
DEFAULT_scale = True
DEFAULT_feature_selection = False
DEFAULT_save = True
DEFAULT_scoring = ''
DEFAULT_project_name = 'polysis'
DEFAULT_concurrency = 1
DEFAULT_verbose = True
DEFAULT_num_degrees = 1
DEFAULT_path = '.'


class Polysis(ABC):
    """Polysis - Abstract baseclass for Polyssifier, Polygressor,
        Polyclusterer, and Polyneuro.

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
                 n_folds=DEFAULT_n_folds, scale=DEFAULT_scale,
                 include=DEFAULT_include,
                 feature_selection=DEFAULT_feature_selection,
                 save=DEFAULT_save, scoring=DEFAULT_scoring,
                 project_name=DEFAULT_project_name,
                 concurrency=DEFAULT_concurrency, verbose=DEFAULT_verbose,
                 num_degrees=DEFAULT_num_degrees, path=DEFAULT_path):
        assert label.shape[0] == data.shape[0],\
            "Label dimesions do not match data number of rows"
        self.data = data
        self.label = label
        self.n_class = len(np.unique(label))
        self.n_folds = n_folds
        self.scale = scale
        self.include = include
        self.feature_selection = feature_selection
        self.save = save
        self.scoring = scoring
        self.project_name = project_name
        self.concurrency = concurrency
        self.verbose = verbose
        self.data = data
        self.num_degrees = num_degrees
        self.path = path
        self.confusions = {}
        self.coefficients = {}

    @abstractmethod
    def initialize_folds(self):
        pass

    @abstractmethod
    def initialize_models(self):
        pass

    @abstractmethod
    def gather_results(self):
        pass

    @abstractmethod
    def feature_importance(self, clf):
        pass

    @abstractmethod
    def _scorer(self, clf, X, y):
        pass

    @abstractmethod
    def aggregate_results(self):
        pass

    def create_project(self):
        if self.save and not os.path.exists(os.path.join(self.path, '{}/models').format(self.project_name)):
            os.makedirs(os.path.join(
                self.path, '{}/models').format(self.project_name))

    def initialize_scores(self):
        logger.info('Initializing scores...')
        self.scores = pd.DataFrame(columns=pd.MultiIndex.from_product(
            [self.models.keys(), ['train', 'test']]),
            index=range(self.n_folds))

    def initialize_predictions(self):
        logger.info('Initializing predictions...')
        self.predictions = pd.DataFrame(columns=self.models.keys(),
                                        index=range(self.data.shape[0]))

    def initialize_probabilities(self):
        logger.info('Initializing test probabilities...')
        self.test_prob = pd.DataFrame(columns=self.models.keys(),
                                      index=range(self.data.shape[0]))

    def finalize_folds(self):
        self.stratified_k_fold.get_n_splits(np.zeros(self.data.shape[0]),
                                            self.label)
        self.k_fold = list(self.stratified_k_fold.split(
                           np.zeros(self.data.shape[0]), self.label))

    def build(self):
        self.initialize_models()
        self.initialize_scores()
        self.initialize_predictions()
        self.initialize_probabilities()
        self.initialize_folds()
        self.finalize_folds()
        # Arg storage
        logger.info('Storing input...')
        self.manager = Manager()
        self.process_args()
        logger.info('Building complete..')

    def run(self):
        logger.info('Running polyssifier.')

        # Parallel Processing of tasks
        logger.info('Fitting models...')
        self.fit_all_models()
        # Gather results
        logger.info('Gathering model results...')
        self.fitted_models = {key: [] for key in self.models}
        self.gather_results()
        # Aggregate results
        self.aggregate_results()
        self.generate_report()
        self.save_results()
        self.print_scores()

    def get_xy(self, args, indices):
        X = args[0]['X'][indices, :]
        y = args[0]['y'][indices]
        return X, y

    def fit_model(self, args, name, val, n_fold, project_name, save, scoring):
        start = time.time()
        train, test = args[0]['k_fold'][n_fold]
        clf = deepcopy(val['clf'])
        if val['parameters']:
            clf = GridSearchCV(clf, val['parameters'], n_jobs=1, cv=3,
                               scoring=self._scorer)

        logger.info('Training {} {}'.format(name, n_fold))
        X, y = self.get_xy(args, train)
        clf.fit(X, y)
        train_score = self._scorer(clf, X, y)

        logger.info('Testing {} {}'.format(name, n_fold))
        X, y = self.get_xy(args, test)
        test_score = self._scorer(clf, X, y)
        ypred = clf.predict(X)
        yprob = 0
        duration = time.time() - start

        logger.info('{0:25} {1:2}: Train {2:.2f}/Test {3:.2f}, {4:.2f} sec'.format(
            name, n_fold, train_score, test_score, duration))
        # Feature importance
        coefficients = self.feature_importance(clf)
        return (train_score, test_score,
                ypred, yprob,  # predictions and probabilities
                coefficients,  # Coefficients for feature ranking
                clf)  # fitted clf

    def generate_report(self):
        # Report generation
        logger.info('Generating report...')
        self.report = Report(scores=self.scores, confusions=self.confusions,
                             predictions=self.predictions,
                             test_prob=self.test_prob, scoring=self.scoring,
                             coefficients=self.coefficients,
                             feature_selection=self.feature_selection, save=self.save)
        self.format_report_summary()

    def save_results(self):
        if self.save:
            logger.info('Saving...')
            self.save_object('poly_' + self.project_name +
                             '/scores.pkl', self.scores)
            self.save_object('poly_' + self.project_name +
                             '/report.pkl', self.report)

    def print_scores(self):
        if self.verbose:
            print(self.scores.astype('float').describe().transpose()
                  [['mean', 'std', 'min', 'max']])

    def process_args(self):
        '''
            Process input, and store in two variables:
            args - just the full args
            args2 - arguments related to input models
        '''
        self.args = self.manager.list()
        self.args.append({})  # Store inputs
        self.args2 = []
        for model_name, val in self.models.items():
            for n_fold in range(self.n_folds):
                self.args2.append((self.args, model_name, val, n_fold,
                                   self.project_name,
                                   self.save, self.scoring))
        self.shared = self.args[0]
        self.shared['k_fold'] = self.k_fold
        self.shared['X'] = self.data
        self.shared['y'] = self.label
        self.args[0] = self.shared

    def fit_all_models(self):
        '''
            Initialize parallel pool
        '''
        proc = self.fit_model
        if self.concurrency == 1:
            self.result = list(starmap(proc, self.args2))
        else:
            pool = Pool(processes=self.concurrency)
            self.result = pool.starmap(proc, self.args2)
            pool.close()

    def _save_object(self, fname, obj):
        '''dump object to pickle'''
        with open(fname, 'wb') as f:
            p.dump(obj, f, protocol=2)

    def create_polynomial(self, data, degree):
        '''
        :param data: the data (numpy matrix) which will have its data vectors raised to powers
        :param degree: the degree of the polynomial we wish to predict
        :return: a new data matrix of the specified degree (for polynomial fitting purposes)
        '''

        # First we make an empty matrix which is the size of what we wish to pass through to linear regress
        height_of_pass_through = data.shape[0]
        width_of_pass_through = degree * data.shape[1]
        to_pass_through = np.zeros(
            shape=(height_of_pass_through, width_of_pass_through))

        # These are the width and height of each "exponeneted" matrix
        height_exponential_matrix = data.shape[0]
        width_exponential_matrix = data.shape[1]

        for i in range(degree):
            to_add_in = data ** (i + 1)
            for j in range(height_exponential_matrix):
                for k in range(width_exponential_matrix):
                    to_pass_through.itemset(
                        (j, k + i * width_exponential_matrix), (to_add_in.item(j, k)))
        return to_pass_through

    def format_report_summary(self):
        self.scores.columns = [' '.join(c) for c in self.scores.columns]
        train_cols = [col for col in self.scores.columns if 'train' in col]
        test_cols = [col for col in self.scores.columns if 'test' in col]
        train_scores = self.scores[train_cols]
        test_scores = self.scores[test_cols]
        headers = [['', 'Mean %s' % self.scoring, 'std %s' %
                    self.scoring, 'Min %s' % self.scoring, 'Max %s' % self.scoring]]
        train_summary = train_scores.astype('float').describe().transpose()[
            ['mean', 'std', 'min', 'max']]
        train_summary = train_summary.sort_values('mean', ascending=False)
        train_summary = train_summary.round(4)

        test_summary = test_scores.astype('float').describe().transpose()[
            ['mean', 'std', 'min', 'max']]
        test_summary = test_summary.sort_values('mean', ascending=False)
        test_summary = test_summary.round(4)

        self.report.summary = {0: headers+[list(s) for s in train_summary.itertuples()]
                               + [['', '', '', '', '']]+[list(s) for s in test_summary.itertuples()]}

        self.report.train_summary = {
            0: headers + [list(s) for s in train_summary.itertuples()]}
        self.report.test_summary = {
            0: headers+[list(s) for s in test_summary.itertuples()]}
