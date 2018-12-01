#! /usr/bin/env python
import sys
import numpy as np
import pickle as p
from multiprocessing import Manager, Pool
import os
import pandas as pd
from abc import 
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
sys.setrecursionlimit(10000)
logger = make_logger('polyssifier')

PERMITTED_SCORINGS = {'classifier': ['auc'], 'regress': ['r2', 'mse']}
DEFAULT_do_regress=False
DEFAULT_n_folds = 10
DEFAULT_scale = True
DEFAULT_exclude = []
DEFAULT_feature_selection = False
DEFAULT_save = False
DEFAULT_scoring = 'auc'
DEFAULT_project_name=''
DEFAULT_concurrency = 1
DEFAULT_verbose = True
DEFAULT_num_degrees = 1

class Polysis():
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
                 n_folds=DEFAULT_n_folds, scale=DEFAULT_scale,
                 exclude=DEFAULT_exclude,
                 feature_selection=DEFAULT_feature_selection,
                 save=DEFAULT_save, scoring=DEFAULT_scoring,
                 project_name=DEFAULT_project_name,
                 concurrency=DEFAULT_concurrency, verbose=DEFAULT_verbose,
                 num_degrees=DEFAULT_num_degrees):
        assert label.shape[0] == data.shape[0],\
        "Label dimesions do not match data number of rows"
        self.data = data
        self.label = label
        self.n_class = len(np.unique(label))
        self.n_folds = n_folds
        self.scale = scale
        self.exclude = exclude
        self.feature_selection = feature_selection
        self.save = save
        self.scoring = scoring
        self.project_name = project_name
        self.concurrency = concurrency
        self.verbose = verbose
        self.data = data
        self.num_degrees = num_degrees
        self.confusions = {}
        self.coefficients = {}
        if save and not os.path.exists('poly_{}/models'.format(project_name)):
            os.makedirs('poly_{}/models'.format(project_name))
        if self.do_classify and self.scoring not in PERMITTED_SCORINGS['classifier']:
            self.scoring = PERMITTED_SCORINGS['classifier'][0]
        if self.do_regress and self.scoring not in PERMITTED_SCORINGS['regress']:
            self.scoring = PERMITTED_SCORINGS['regress'][0]

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

    def build(self):
        '''
        Polyssifier.build()
        Builds the models - either classifiers or regressors.
        Initializes scores dataframe.
        Initializes predictions dataframe.
        Initializes test_probability dataframe.
        Initializes CV folds
        Processes and stores input args.
        '''

        self.initialize_scores()
        self.initialize_predictions()
        self.initialize_probabilities()
        
        # CV Fold generation
        logger.info('Initializing CV folds...')
        if self.do_classify:
            self.stratified_k_fold = StratifiedKFold(n_splits=self.n_folds,
                                                         random_state=1988,
                                                         shuffle=True)
        if self.do_regress:
            self.stratified_k_fold = KFold(n_splits=self.n_folds,
                                           random_state=1988)
        self.stratified_k_fold.get_n_splits(np.zeros(self.data.shape[0]),
                                            self.label)
        self.k_fold = list(self.stratified_k_fold.split(
                           np.zeros(self.data.shape[0]), self.label))
        
        # Arg storage
        logger.info('Storing input...')
        self.manager = Manager()
        self.process_args()    
        logger.info('Building complete..')
    
    def run(self):
        '''
            Run the analysis.
            Fits the models, gathers results, aggregates scores,
            generates the report, and saves results.
        '''
        logger.info('Running polyssifier.')

        # Parallel Processing of tasks
        logger.info('Fitting models...')
        self.fit_models()

        # Gather results
        logger.info('Gathering model results...')
        self.fitted_models = {key: [] for key in self.models}
        self.gather_results()
        
        # Aggregated measures (voting or median scores)
        if self.do_classify:
            logger.info('Performing classifier voting...')
            self.perform_voting()
        if self.do_regress:
            logger.info('Computing regression median scores...')
            self.compute_median()

        # Report generation
        logger.info('Generating report...')
        self.report = Report(scores=self.scores, confusions=self.confusions,
                             predictions=self.predictions,
                             test_prob=self.test_prob, scoring=self.scoring,
                             coefficients=self.coefficients,
                             feature_selection=self.feature_selection)            
        
        # Saving
        if self.save:
            logger.info('Saving...')
            if self.do_classify:
                self.save_object('poly_' + self.project_name +
                          '/confusions.pkl', self.confusions)
            self.save_object('poly_' + self.project_name +
                             '/scores.pkl', self.scores)
            self.save_object('poly_' + self.project_name +
                             '/report.pkl', self.report)
        # Print scores
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

    def fit_models(self):
        '''
            Initialize parallel pool
        '''
        proc = self.fit_reg if self.do_regress else self.fit_clf
        if self.concurrency == 1:
            self.result = list(starmap(proc, self.args2))
        else:
            pool = Pool(processes=self.concurrency)
            self.result = pool.starmap(proc, self.args2)
            pool.close()

    def gather_results(self):
        '''Gathers results'''
        for clf_name in self.models:
            self.coefficients[clf_name] = []
            temp = np.zeros((self.n_class, self.n_class))
            temp_pred = np.zeros((self.data.shape[0], ))
            temp_prob = np.zeros((self.data.shape[0], ))
            clfs = self.fitted_models[clf_name]
            for n in range(self.n_folds):
                if self.do_classify:
                    train_score, test_score, prediction, prob, confusion,\
                        coefs, fitted_model = self.result.pop(0)
                if self.do_regress:
                    train_score, test_score, prediction, prob,\
                        coefs, fitted_model = self.result.pop(0)
                clfs.append(fitted_model)
                self.scores.loc[n, (clf_name, 'train')] = train_score
                self.scores.loc[n, (clf_name, 'test')] = test_score
                if self.do_classify:
                    temp += confusion
                temp_prob[self.k_fold[n][1]] = prob
                if self.do_classify:
                    temp_pred[self.k_fold[n][1]] = self._le.inverse_transform(prediction)
                if self.do_regress:
                    temp_pred[self.k_fold[n][1]] = prediction
                self.coefficients[clf_name].append(coefs)
            if self.do_classify:
                self.confusions[clf_name] = temp
            self.predictions[clf_name] = temp_pred
            self.test_prob[clf_name] = temp_prob

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

    def compute_median(self):
        self.fitted_models = pd.DataFrame(self.fitted_models)
        self.scores['Median', 'train'] = np.zeros((self.n_folds, ))
        self.scores['Median', 'test'] = np.zeros((self.n_folds, ))
        temp_pred = np.zeros((self.data.shape[0], ))
        for n, (train, test) in enumerate(self.k_fold):
            reg = MyRegressionMedianer(self.fitted_models.loc[n].values)
            X, y = self.data[train, :], self.label[train]
            self.scores.loc[n, ('Median', 'train')] = self._reg_scorer(reg, X,
                            y, self.scoring)
            X, y = self.data[test, :], self.label[test]
            self.scores.loc[n, ('Median', 'test')] = self._reg_scorer(reg, X,
                            y, self.scoring)
            temp_pred[test] = reg.predict(X)
        self.predictions['Median'] = temp_pred

    def _save_object(self, fname, obj):
        '''dump object to pickle'''
        with open(fname, 'wb') as f:
                p.dump(obj, f, protocol=2)

    def _scorer(self, clf, X, y):
        '''Function that scores a classifier according to what is available as a
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
                    print('predict proba return shape{}'.format(ypred.shape))
    
                assert len(ypred.shape) == 1,\
                    'predict proba return shape {}'.format(ypred.shape)
            elif hasattr(clf, 'decision_function'):
                ypred = clf.decision_function(X)
                assert len(ypred.shape) == 1,\
                    'decision_function return shape {}'.format(ypred.shape)
            else:
                ypred = clf.predict(X)
            score = roc_auc_score(y, ypred)
        else:
            score = f1_score(y, clf.predict(X))
        return score

    def fit_clf(self, args, clf_name, val, n_fold, project_name, save, scoring):
        '''
        Multiprocess safe function that fits classifiers
        args: shared dictionary that contains
            - X = all data
            - y = all labels
            - k_fold = list of train and test indexes for each fold
            - clf_name = name of the classifier model
            - val = dictionary with
            - clf = sklearn compatible classifier
            - parameters = dictionary with parameters, can be used for grid search
            - n_fold = number of folds
            - project_name = string with the project folder name to save model
        '''
        train, test = args[0]['k_fold'][n_fold]
        X = args[0]['X'][train, :]
        y = args[0]['y'][train]
        file_name = 'poly_{}/models/{}_{}.p'.format(
            project_name, clf_name, n_fold + 1)
        start = time.time()
        if save and os.path.isfile(file_name):
            logger.info('Loading {} {}'.format(file_name, n_fold))
            clf = joblib.load(file_name)
        else:
            logger.info('Training {} {}'.format(clf_name, n_fold))
            clf = deepcopy(val['clf'])
            if val['parameters']:
                clf = GridSearchCV(clf, val['parameters'], n_jobs=1, cv=3,
                                   scoring=self._scorer)
            clf.fit(X, y)
            if save:
                joblib.dump(clf, file_name)
    
        train_score = self._scorer(clf, X, y)
    
        X = args[0]['X'][test, :]
        y = args[0]['y'][test]
        # Scores
        test_score = self._scorer(clf, X, y)
        ypred = clf.predict(X)
        yprob = None
        if hasattr(clf, 'predict_proba'):
            # For compatibility with different sklearn versions
            yprob = clf.predict_proba(X)
            try:
                yprob = yprob[:, 1]
            except:
                print('predict proba return shape {}'.format(yprob.shape))
    
        elif hasattr(clf, 'decision_function'):
            yprob = clf.decision_function(X)
            try:
                yprob = yprob[:, 1]
            except:
                print('predict proba return shape {}'.format(yprob.shape))
    
            assert len(yprob.shape) == 1,\
                'predict proba return shape {}'.format(ypred.shape)
    
        confusion = confusion_matrix(y, ypred)
        duration = time.time() - start
        logger.info('{0:25} {1:2}: Train {2:.2f}/Test {3:.2f}, {4:.2f} sec'.format(
            clf_name, n_fold, train_score, test_score, duration))
    
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
    
        return (train_score, test_score,
                ypred, yprob,  # predictions and probabilities
                confusion,  # confusion matrix
                coefficients,  # Coefficients for feature ranking
                clf)  # fitted clf
    
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

    def _reg_scorer(self, reg, X, y, scoring):
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
        if scoring == 'mse':
            return mean_squared_error(y, reg.predict(X))
        else:
            return r2_score(y, reg.predict(X))
    
    
    def fit_reg(self, args, reg_name, val, n_fold, project_name, save, scoring):
        '''
        Multiprocess safe function that fits classifiers
        args: shared dictionary that contains
            - X = all data
            - y = all labels
            - k_fold = list of train and test indexes for each fold
            - reg_name = name of the classifier or regressor model
            - val = dictionary with
            - reg = sklearn compatible classifier 
            - parameters = dictionary with parameters, can be used for grid search
            - n_fold = number of folds
            - project_name = string with the project folder name to save model
        '''
    
        # Creates the scoring string to pass into grid search.
        if scoring == 'mse':
            scorestring = 'neg_mean_squared_error'
        elif scoring == 'r2':
            scorestring = 'r2'
        else:
            scorestring = 'r2'
    
        train, test = args[0]['k_fold'][n_fold]
        X = args[0]['X'][train, :]
        y = args[0]['y'][train]
        file_name = 'polyr_{}/models/{}_{}.p'.format(
            project_name, reg_name, n_fold + 1)
        start = time.time()
        if os.path.isfile(file_name):
            logger.info('Loading {} {}'.format(file_name, n_fold))
            reg = joblib.load(file_name)
        else:
            logger.info('Training {} {}'.format(reg_name, n_fold))
            reg = deepcopy(val['reg'])
            if val['parameters']:
                kfold = KFold(n_splits=3, random_state=1988)
                reg = GridSearchCV(reg, val['parameters'], n_jobs=1, cv=kfold,
                                   scoring=scorestring)
            reg.fit(X, y)
            if save:
                joblib.dump(reg, file_name)
    
        train_score = self._reg_scorer(reg, X, y, scoring)
    
        X = args[0]['X'][test, :]
        y = args[0]['y'][test]
        # Scores
        test_score = self._reg_scorer(reg, X, y, scoring)
        ypred = reg.predict(X)
        yprob = 0
    
        duration = time.time() - start
        logger.info('{0:25} {1:2}: Train {2:.2f}/Test {3:.2f}, {4:.2f} sec'.format(
            reg_name, n_fold, train_score, test_score, duration))
    
        # Feature importance
        if hasattr(reg, 'steps'):
            temp = reg.steps[-1][1]
        elif hasattr(reg, 'best_estimator_'):
            if hasattr(reg.best_estimator_, 'steps'):
                temp = reg.best_estimator_.steps[-1][1]
            else:
                temp = reg.best_estimator_
        try:
            if hasattr(temp, 'coef_'):
                coefficients = temp.coef_
            elif hasattr(temp, 'feature_importances_'):
                coefficients = temp.feature_importances_
            else:
                coefficients = None
        except:
            coefficients = None
    
        return (train_score, test_score,
                ypred, yprob,  # predictions and probabilities
                coefficients,  # Coefficients for feature ranking
                reg)  # fitted reg



