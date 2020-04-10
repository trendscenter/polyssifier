DEFAULT_parameters = {
    'Multilayer Perceptron': {
        'hidden_layer_sizes': [(128,), ],
        'activation': ['relu', 'identity', 'logistic', 'tanh'],
        'solver': ['adam', 'lbfgs', 'sgd'],
        'alpha': [1e-3, 1e-2, 1e-1, 1e-0],
        'batch_size': ['auto'],
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'learning_rate_init': [1e-3, 1e-2, 1e-1],
        'power_t': [0.5],
        'max_iter': [100],
        'shuffle': [True],
        'tol': [1e-6],
        'momentum': [0.9],
        'nesterovs_momentum': [True],
        'early_stopping': [True],
        'validation_fraction': [0.1],
        'beta_1': [0.9],
        'beta_2': [0.999],
        'epsilon': [1e-8],
        'n_iter_no_change': [10],
        'max_fun': [15000]
    },
    'Nearest Neighbors': {
        'n_neighbors': list(range(2, 10)),
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size': [30],
        'p': [2],
        'metric': ['minkowski']
    },
    'Radius Neighbors': {
        'radius': [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size': [30],
        'p': [2],
        'metric': ['minkowski'],

    },
    'SVM': {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'C': [0.01, 0.1, 1],
        'degree': [1, 2, 3, 4],
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
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 4],
        'min_samples_leaf': [1, 2],
        'min_weight_fraction_leaf': [0],
        'max_features': [None, 'auto', 'sqrt', 'log2'],
        'max_leaf_nodes': [None],
        'min_impurity_decrease': [0],
    },
    'Random Forest': {
        'n_estimators': [250, 500],
        'criterion': ['entropy', 'gini'],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'min_weight_fraction_leaf': [0],
        'max_features': [None, 'auto', 'sqrt', 'log2'],
        'max_leaf_nodes': [None],
        'min_impurity_decrease': [0],
    },
    'Ada Boost': {
        'base_estimator': [None],
        'learning_rate': [1, 1e-1, 1e-2],
        'algorithm': ['SAMME.R', 'SAMME'],
    },
    'Bagging': {
        'base_estimator': [None],
        'n_estimators': [100, 250],
        'max_samples': [1.0],
        'max_features': [1],
    },
    'Extra Trees': {
        'n_estimators': [100, 250],
        'criterion': ['gini', 'entropy'],
        'max_depth': [None],
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'min_weight_fraction_leaf': [0],
        'max_features': [None, 'auto', 'sqrt', 'log2'],
        'max_leaf_nodes': [None],
        'min_impurity_decrease': [0],
    },
    'Gradient Boost': {
        'loss': ['deviance', 'exponential'],
        'learning_rate': [1e-0, 1e-1, 1e-2, 1e-3],
        'n_estimators': [100, 250],
        'subsample': [1],
        'criterion': ['friedman_mse', 'mse', 'mae']
    },
    'Logistic Regression': {
        'penalty': ['l2', 'l1', 'elasticnet', 'none'],
        'tol': [1e-6],
        'C': [0.001, 0.01, 0.1, 1],
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        'max_iter': [500],
    },
    'Ridge Classification': {
        'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
    },
    'Passive Aggressive': {
        'C': [0.001, 0.01, 0.1, 1.0],
        'max_iter': [1000],
        'tol': [1e-6],
        'loss': ['hinge', 'squared_hinge']
    },
    'Perceptron': {
        'penalty': [None, 'l2', 'l1', 'elasticnet'],
        'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
        'max_iter': [1000],
        'eta0': [1],
    },
    'Guassian Naive Bayes': {
        'priors': [None],
        'var_smoothing': [1e-09]
    },
    'Bernoulli Naive Bayes': {
        'alpha': [1, 1e-1, 1e-2],
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
        'max_iter_predict': [500]
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
