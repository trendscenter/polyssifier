DEFAULT_parameters = {
    'Multilayer Perceptron': {
        'hidden_layer_sizes': [(100,)],
        'activation': ['relu', 'identity', 'logistic', 'tanh'],
        'solver': ['adam', 'lbfgs', 'sgd'],
        'alpha': [0.0001],
        'batch_size': ['auto'],
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'learning_rate_init': [0.001],
        'power_t': [0.5],
        'max_iter': [500],
        'shuffle': [True],
        'tol': [1e-4],
        'momentum': [0.9],
        'nesterovs_momentum': [True],
        'early_stopping': [False],
        'validation_fraction': [0.1],
        'beta_1': [0.9],
        'beta_2': [0.999],
        'epsilon': [1e-8],
        'n_iter_no_change': [10],
        'max_fun': [15000]
    },
    'Nearest Neighbors': {
        'n_neighbors': [5],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size': [30],
        'p': [2],
        'metric': ['minkowski']
    },
    'Radius Neighbors': {
        'radius': [1.0],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size': [30],
        'p': [2],
        'metric': ['minkowski'],

    },
    'SVM': {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'C': [0.01, 0.1, 1],
        'degree': [3],
        'gamma': ['scale', 'auto'],
        'coef0': [0],
        'shrinking': [True],
        'tol': [1e-3],
        'max_iter': [-1],
        'decision_function_shape': ['ovr', 'ovo']
    },
    'Decision Tree': {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': [None],
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'min_weight_fraction_leaf': [0],
        'max_features': [None, 'auto', 'sqrt', 'log2'],
        'max_leaf_nodes': [None],
        'min_impurity_decrease': [0],
        'min_impurity_split': [1e-7],
    },
    'Random Forest': {
        'n_estimators': list(range(5, 20)),
        'criterion': ['entropy', 'gini'],
        'max_depth': [None],
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'min_weight_fraction_leaf': [0],
        'max_features': [None, 'auto', 'sqrt', 'log2'],
        'max_leaf_nodes': [None],
        'min_impurity_decrease': [0],
        'min_impurity_split': [1e-7],
    },
    'Ada Boost': {
        'base_estimator': [None],
        'learning_rate': [1],
        'algorithm': ['SAMME.R', 'SAMME'],
    },
    'Bagging': {
        'base_estimator': [None],
        'n_estimators': [10],
        'max_samples': [1.0],
        'max_features': [1],
    },
    'Extra Trees': {
        'n_estimators': [100],
        'criterion': ['gini', 'entropy'],
        'max_depth': [None],
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'min_weight_fraction_leaf': [0],
        'max_features': [None, 'auto', 'sqrt', 'log2'],
        'max_leaf_nodes': [None],
        'min_impurity_decrease': [0],
        'min_impurity_split': [1e-7],
    },
    'Gradient Boost': {
        'loss': ['deviance', 'exponential'],
        'learning_rate': [0.1],
        'n_estimators': [100],
        'subsample': [1],
        'criterion': ['friedman_mse', 'mse', 'mae']
    },
    'Logistic Regression': {
        'penalty': ['l2', 'l1', 'elasticnet', 'none'],
        'tol': [1e-4],
        'C': [0.001, 0.1, 1],
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        'max_iter': [100],
    },
    'Ridge Classification CV': {
        'alphas': [0.1, 1.0, 10.0]
    },
    'Passive Aggressive': {
        'C': [1.0],
        'max_iter': [1000],
        'tol': [0.001],
        'loss': ['hinge', 'squared_hinge']
    },
    'Perceptron': {
        'penalty': [None, 'l2', 'l1', 'elasticnet'],
        'alpha': [0.0001],
        'max_iter': [1000],
        'eta0': [1],
    },
    'Guassian Naive Bayes': {
        'priors': [None],
        'var_smoothing': [1e-09]
    },
    'Bernoulli Naive Bayes': {
        'alpha': [1],
        'binarize': [0],
    },
    'Categorical Naive Bayes': {
        'alpha': [1]
    },
    'Multinomail Naive Bayes': {
        'alpha': [1]
    },
    'Gaussian Process': {
        'kernel': [None],
        'optimizer': ['fmin_l_bfgs_b'],
        'max_iter_predict': [100]
    },
    'Linear Regression': {},
    'Bayesian Ridge': {},
    'PassiveAggressiveRegressor': {},
    'GaussianProcessRegressor': {},
    'Ridge': {},
    'Lasso': {},
    'Lars': {},
    'LassoLars': {},
    'OrthogonalMatchingPursuit': {},
    'ElasticNet': {},
    'ARD Regression': {},
    'Huber Regression': {},
    'RANSAC Regression': {},
    'SGD Regression': {},
    'TheilSen Regression': {},
    'MLP Regression': {},
    'KNN Regression': {}
}
