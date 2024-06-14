from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error
from data import dataprep_linear, dataprep_ridge, dataprep_lasso, dataprep_xgboost
from xgboost import XGBRegressor


# Configurations and hyperparameters
SEED = 42       # random seed for reproducibility
TRAIN_SPLIT = 0.8     # training size (in fraction)

# Evaluation metrics
METRICS = {
    'r2': r2_score,
    'mae':  mean_absolute_error
}

# Models
# list of supported models, their dataprep functions (defined in data\custom_dattaprep.py), whether to enable log transformation, and the hyper-parameters to optimze
# you can add more by following the strcture outlined here
ARGS_DICT = {
        'linear':                   # command line arg
            ['linear',              # model name
             LinearRegression,      # model
             {},                    # model parameters
             dataprep_linear,       # dataprep function
             True,                  # log transformation
             None                   # hyperparameters to optimize
             ],
        
        'ridge':                    # command line arg
            ['ridge',               # model name
             Ridge,                 # model
             {},                    # model parameters
             dataprep_ridge,        # dataprep function
             True,                  # log transformation
             {                      # hyperparameters to optimize
                 'alpha': lambda trial: trial.suggest_categorical('alpha', [0.1, 0.3, 0.5, 0.7, 1.0, 2, 4, 8, 16, 32])
             }],

        'lasso':                    # command line arg
            ['lasso',               # model name
             Lasso,                 # model
             {},                    # model parameters
             dataprep_lasso,        # dataprep function
             False,                 # log transformation
             {                      # hyperparameters to optimize
                 'alpha': lambda trial: trial.suggest_categorical('alpha', [0.1, 0.3, 0.5, 0.7, 1.0, 2, 4, 8, 16, 32])
             }],

        'xgboost':                                                  # command line arg
            ['xgboost',                                             # model name
             XGBRegressor,                                          # model
             {'enable_categorical' : True, 'random_state' : 42},    # model parameters  
             dataprep_xgboost,                                      # dataprep function
             False,                                                 # log transformation
             {                                                      # hyperparameters to optimize
                'lambda': lambda trial: trial.suggest_float('lambda', 1e-8, 1.0, log=True),
                'alpha': lambda trial: trial.suggest_float('alpha', 1e-8, 1.0, log=True),
                'colsample_bytree': lambda trial: trial.suggest_categorical('colsample_bytree', [0.3, 0.4, 0.5, 0.7]),
                'subsample': lambda trial: trial.suggest_categorical('subsample', [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
                'learning_rate': lambda trial: trial.suggest_float('learning_rate', 1e-8, 1.0, log=True),
                'n_estimators': lambda trial: trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': lambda trial: trial.suggest_int('max_depth', 3, 9),
                'random_state': 42,
                'min_child_weight': lambda trial: trial.suggest_int('min_child_weight', 1, 10),
                'enable_categorical': True
             }],
    }