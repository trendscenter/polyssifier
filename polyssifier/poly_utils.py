from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              BaggingClassifier, GradientBoostingClassifier)

from sklearn.linear_model import (LogisticRegression,
                                  LinearRegression,
                                  BayesianRidge, Ridge,
                                  RidgeClassifier, Lasso,
                                  ElasticNet, Lars, LassoLars,
                                  OrthogonalMatchingPursuit,
                                  PassiveAggressiveRegressor,
                                  PassiveAggressiveClassifier,
                                  SGDClassifier, Perceptron,
                                  ARDRegression, HuberRegressor,
                                  RANSACRegressor, SGDRegressor,
                                  TheilSenRegressor,
                                  ElasticNetCV,  LarsCV, LassoCV,
                                  LassoLarsCV, LogisticRegressionCV,
                                  OrthogonalMatchingPursuitCV, RidgeCV,
                                  RidgeClassifierCV)
from sklearn.naive_bayes import (GaussianNB, MultinomialNB, BernoulliNB)
from sklearn.neural_network import MLPClassifier, MLPRegressor, BernoulliRBM
from sklearn.gaussian_process import (GaussianProcessRegressor,
                                      GaussianProcessClassifier)
from sklearn.neighbors import (KNeighborsClassifier, RadiusNeighborsClassifier,
                               NearestCentroid, KNeighborsRegressor,
                               RadiusNeighborsRegressor)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import collections
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process.kernels import RBF


class MyVoter(object):
    """
    Voter Classifier
    Receives fitted classifiers and runs majority voting
    """

    def __init__(self, estimators):
        '''
        estimators: List of fitted classifiers
        '''
        self.estimators_ = estimators

    def predict(self, X):
        predictions = np.asarray(
            [clf.predict(X) for clf in self.estimators_]).T
        maj = np.apply_along_axis(
            lambda x: np.argmax(np.bincount(x)), axis=1,
            arr=predictions.astype('int'))
        return maj


class MyRegressionAverager(object):
    """
    Regression averager
    Receives fitted regressors and averages the predictions of the regressors.
    """

    def __init__(self, estimators):
        '''
        estimators: List of fitted regressors
        '''
        self.estimators_ = estimators

    def predict(self, X):
        predictions = np.asarray(
            [reg.predict(X) for reg in self.estimators_]).T

        avg = np.average(predictions, axis=1)
        return avg


class MyRegressionMedianer(object):
    """
    Regression averager
    Receives fitted regressors and averages the predictions of the regressors.
    """

    def __init__(self, estimators):
        '''
        estimators: List of fitted regressors
        '''
        self.estimators_ = estimators

    def predict(self, X):
        predictions = np.asarray(
            [reg.predict(X) for reg in self.estimators_]).T

        avg = np.median(predictions, axis=1)
        return avg


def build_classifiers(include, scale, feature_selection, nCols):
    '''
    Input:
        - include: list of names of classifiers to include from the analysis
        - scale: True or False. Scale data before fitting classifier
        - feature_selection: True or False. Run feature selection before
        fitting classifier
        - nCols: Number of columns in input dataset to classifiers
    Output:
    Dictionary with classifier name as keys.
        - 'clf': Classifier object
        - 'parameters': Dictionary with parameters of 'clf' as keys
    '''
    classifiers = collections.OrderedDict()

    '''Neural Networks'''
    if 'Multilayer Perceptron' in include:
        classifiers['Multilayer Perceptron'] = {
            'clf': MLPClassifier(),
            'parameters': {'hidden_layer_sizes': [(100, 50), (50, 25)],
                           'max_iter': [500]}
        }

    '''Neighbor Methods'''
    if 'Nearest Neighbors' in include:
        classifiers['Nearest Neighbors'] = {
            'clf': KNeighborsClassifier(),
            'parameters': {'n_neighbors': [1, 5, 10, 20]}}

    if 'Radius Neighbors' in include:
        classifiers['Radius Neighbors'] = {
            'clf': RadiusNeighborsClassifier(outlier_label=0),
            'parameters': {}}

    '''SVM'''
    if 'SVM' in include:
        classifiers['SVM'] = {
            'clf': SVC(C=1, probability=True, cache_size=10000,
                       class_weight='balanced'),
            'parameters': {'kernel': ['rbf', 'poly'],
                           'C': [0.01, 0.1, 1]}}

    if 'Linear SVM' in include:
        classifiers['Linear SVM'] = {
            'clf': LinearSVC(dual=False, class_weight='balanced'),
            'parameters': {'C': [0.01, 0.1, 1],
                           'penalty': ['l1', 'l2']}}

    '''Tree Methods'''
    if 'Decision Tree' in include:
        classifiers['Decision Tree'] = {
            'clf': DecisionTreeClassifier(max_depth=None,
                                          max_features='auto'),
            'parameters': {}}

    if 'Random Forest' in include:
        classifiers['Random Forest'] = {
            'clf': RandomForestClassifier(max_depth=None,
                                          n_estimators=10,
                                          max_features='auto'),
            'parameters': {'n_estimators': list(range(5, 20))}}

    '''Ensemble Methods'''
    if 'Ada Boost' in include:
        classifiers['Ada Boost'] = {
            'clf': AdaBoostClassifier(),
            'parameters': {}}

    if 'Bagging' in include:
        classifiers['Ada Boost'] = {
            'clf': BaggingClassifier(),
            'parameters': {}}

    if 'Gradient Boost' in include:
        classifiers['Gradient Boost'] = {
            'clf': GradientBoostingClassifier(),
            'parameters': {}}

    ''' Linear Models '''
    if 'Logistic Regression' in include:
        classifiers['Logistic Regression'] = {
            'clf': LogisticRegression(fit_intercept=True, solver='lbfgs',
                                      penalty='l2'),
            'parameters': {'C': [0.001, 0.1, 1]}}

    if 'Ridge Classification' in include:
        classifiers['Ridge Classification'] = {
            'clf': RidgeClassifier(fit_intercept=True),
            'parameters': {}}

    if 'Ridge Classification CV' in include:
        classifiers['Ridge Classification CV'] = {
            'clf': RidgeClassifierCV(fit_intercept=True),
            'parameters': {}}

    if 'Passive Aggressive' in include:
        classifiers['Passive Aggressive Classifier'] = {
            'clf': PassiveAggressiveClassifier(),
            'parameters': {}}

    if 'Perceptron' in include:
        classifiers['Perceptron'] = {
            'clf': Perceptron(),
            'parameters': {}}

    '''Naive Bayes'''
    if 'Gaussian Naive Bayes' in include:
        classifiers['Gaussian Naive Bayes'] = {
            'clf': GaussianNB(),
            'parameters': {}}

    if 'Bernoulli Naive Bayes' in include:
        classifiers['Bernoulli Naive Bayes'] = {
            'clf': BernoulliNB(),
            'parameters': {}}

    '''Discriminant Analysis'''
    if 'LDA' in include:
        classifiers['LDA'] = {
            'clf': LinearDiscriminantAnalysis(),
            'parameters': {}}

    if 'QDA' in include:
        classifiers['QDA'] = {
            'clf': QuadraticDiscriminantAnalysis(),
            'parameters': {}}

    if 'Gaussian Process' in include:
        classifiers['Guassian Process'] = {
            'clf': GaussianProcessClassifier(),
            'parameters': {}}

    # classifiers['Voting'] = {}

    def name(x):
        """
        :param x: The name of the classifier
        :return: The class of the final estimator in lower case form
        """
        return x['clf']._final_estimator.__class__.__name__.lower()

    for key, val in classifiers.items():
        if not scale and not feature_selection:
            break
        steps = []
        if scale:
            steps.append(StandardScaler())
        if feature_selection:
            steps.append(SelectKBest(f_regression, k='all'))
        steps.append(classifiers[key]['clf'])
        classifiers[key]['clf'] = make_pipeline(*steps)
        # Reorganize paramenter list for grid search
        new_dict = {}
        for keyp in classifiers[key]['parameters']:
            new_dict[name(classifiers[key]) + '__' +
                     keyp] = classifiers[key]['parameters'][keyp]
        classifiers[key]['parameters'] = new_dict
        if nCols > 5 and feature_selection:
            classifiers[key]['parameters']['selectkbest__k'] = np.linspace(
                np.round(nCols / 5), nCols, 5).astype('int').tolist()

    return classifiers


def build_regressors(include, scale, feature_selection, nCols):
    '''
    This method builds an ordered dictionary of regressors, where the key is the name of the
    regressor and the value of each key contains a standard dictionary with two keys itself. The first key called
    'reg' points to the regression object, which is created by scikit learn. The second key called 'parameters'
    points to another regular map containing the parameters which are associated with the particular regression model.
    These parameters are used by grid search in polyssifier.py when finding the best model. If parameters are not
    defined then grid search is not performed on that particular regression model, so the model's default parameters
    are used instead to find the best model for the particular data.
    '''
    regressors = collections.OrderedDict()

    if 'Linear Regression' in include:
        regressors['Linear Regression'] = {
            'clf': LinearRegression(),
            'parameters': {}  # Best to leave default parameters
        }

    if 'Bayesian Ridge' in include:
        regressors['Bayesian Ridge'] = {
            'clf': BayesianRidge(),
            'parameters': {}  # Investigate if alpha and lambda parameters should be changed
        }

    if 'PassiveAggressiveRegressor' in include:
        regressors['PassiveAggressiveRegressor'] = {
            'clf': PassiveAggressiveRegressor(),
            'parameters': {'C': [0.5, 1.0, 1.5]
                           }
        }

    if 'GaussianProcessRegressor' in include:
        regressors['GaussianProcessRegressor'] = {
            'clf': GaussianProcessRegressor(),
            'parameters': {
                'alpha': [0.01, 0.1, 1.0, 10.0],
                'kernel': [RBF(x) for x in [0.01, 1.0, 100.0, 1000.0]],
            }
        }

    if 'Ridge' in include:
        regressors['Ridge'] = {
            'clf': Ridge(),
            'parameters': {
                'alpha': [0.25, 0.50, 0.75, 1.00]
            }
        }

    if 'Lasso' in include:
        regressors['Lasso'] = {
            'clf': Lasso(),
            'parameters': {
                'alpha': [0.25, 0.50, 0.75, 1.00]
            }
        }

    if 'Lars' in include:
        regressors['Lars'] = {
            'clf': Lars(),
            'parameters': {}  # Best to leave the default parameters
        }

    if 'LassoLars' in include:
        regressors['LassoLars'] = {
            'clf': LassoLars(),
            'parameters': {'alpha': [0.25, 0.50, 0.75, 1.00, 10.0]}
        }

    if 'OrthogonalMatchingPursuit' in include:
        regressors['OrthogonalMatchingPursuit'] = {
            'clf': OrthogonalMatchingPursuit(),
            'parameters': {}  # Best to leave default parameters
        }

    if 'ElasticNet' in include:
        regressors['ElasticNet'] = {
            'clf': ElasticNet(),
            'parameters': {'alpha': [0.25, 0.50, 0.75, 1.00],
                           'l1_ratio': [0.25, 0.50, 0.75, 1.00]}
        }

    if 'ARD Regression' in include:
        regressors['ARD Regression'] = {
            'clf': ARDRegression(),
            'parameters': {}  # Best to leave default parameters
        }

    if 'Huber Regression' in include:
        regressors['Huber Regression'] = {
            'clf': HuberRegressor(),
            'parameters': {}  # Best to leave default parameters
        }

    if 'RANSAC Regression' in include:
        regressors['RANSAC Regression'] = {
            'clf': RANSACRegressor(),
            'parameters': {}  # Best to leave default parameters
        }

    if 'SGD Regression' in include:
        regressors['SGD Regression'] = {
            'clf': SGDRegressor(),
            'parameters': {}  # Best to leave default parameters
        }

    if 'TheilSen Regression' in include:
        regressors['TheilSen Regression'] = {
            'clf': TheilSenRegressor(),
            'parameters': {}  # Best to leave default parameters
        }

    if 'MLP Regression' in include:
        regressors['MLP Regression'] = {
            'clf': MLPRegressor(),
            'parameters': {}  # Best to leave default parameters
        }

    if 'KNN Regression' in include:
        regressors['KNN Regression'] = {
            'clf': KNeighborsRegressor(),
            'parameters': {}  # Best to leave default parameters
        }

    def name(x):
        """
        :param x: The name of the regressor
        :return: The class of the final regression estimator in lower case form
        """
        return x['clf']._final_estimator.__class__.__name__.lower()

    for key, val in regressors.items():
        if not scale and not feature_selection:
            break
        steps = []
        if scale:
            steps.append(StandardScaler())
        if feature_selection:
            steps.append(SelectKBest(f_regression, k='all'))
        steps.append(regressors[key]['clf'])
        regressors[key]['clf'] = make_pipeline(*steps)
        # Reorganize paramenter list for grid search
        new_dict = {}
        for keyp in regressors[key]['parameters']:
            new_dict[name(regressors[key]) + '__' +
                     keyp] = regressors[key]['parameters'][keyp]
        regressors[key]['parameters'] = new_dict
        if nCols > 5 and feature_selection:
            regressors[key]['parameters']['selectkbest__k'] = np.linspace(
                np.round(nCols / 5), nCols, 5).astype('int').tolist()

    return regressors
