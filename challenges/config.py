from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error
from data import dataprep_linear, dataprep_ridge, dataprep_lasso, dataprep_xgboost
from xgboost import XGBRegressor
from optuna import trial


# Configurations and hyperparameters
SEED = 42       # random seed for reproducibility
TRAIN_SPLIT = 0.8     # training size (in fraction)

# Evaluation metrics
METRICS = {
    'r2': r2_score,
    'mae':  mean_absolute_error
}

# Models
# list of supported models, their dataprep functions, and the hyper-parameters to optimze
#   (defined below)
ARGS_DICT = {
        'linear':                   # command line arg
            [LinearRegression(),    # model  
             dataprep_linear,       # dataprep function
             None                   # hyperparameters to optimize
             ],
        
        'ridge':                    # command line arg
            [Ridge(),               # model
             dataprep_ridge,        # dataprep function
             {                      # hyperparameters to optimize
                 'alpha': lambda trial: trial.suggest_categorical('alpha', [0.1, 0.3, 0.5, 0.7, 1.0, 2, 4, 8, 16, 32])
             }],

        'lasso':                    # command line arg
            [Lasso(),               # model
             dataprep_lasso,        # dataprep function
             {                      # hyperparameters to optimize
                 'alpha': lambda trial: trial.suggest_categorical('alpha', [0.1, 0.3, 0.5, 0.7, 1.0, 2, 4, 8, 16, 32])
             }],

        'xgboost':                                      # command line arg
            [XGBRegressor(enable_categorical = True,    # model
                           random_state = SEED),  
             dataprep_xgboost,                          # dataprep function
             {                                          # hyperparameters to optimize
                 'lambda': lambda trial: trial.suggest_float('lambda', 1e-8, 1.0, log=True),
                 'alpha': lambda trial: trial.suggest_float('alpha', 1e-8, 1.0, log=True),
                 'colsample_bytree': lambda trial: trial.suggest_categorical('colsample_bytree', [0.3, 0.4, 0.5, 0.7]),
                 'subsample': lambda trial: trial.suggest_categorical('subsample', [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
                 'learning_rate': lambda trial: trial.suggest_float('learning_rate', 1e-8, 1.0, log=True),
                 'n_estimators': lambda trial: trial.suggest_int('n_estimators', 100, 1000),
                 'max_depth': lambda trial: trial.suggest_int('max_depth', 3, 9),
                 'random_state': SEED,
                 'min_child_weight': lambda trial: trial.suggest_int('min_child_weight', 1, 10),
                 'enable_categorical': True
             }],
    }   