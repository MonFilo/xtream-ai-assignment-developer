from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error
from data import dataprep_linear, dataprep_xgboost
from xgboost import XGBRegressor


# Configurations and hyperparameters
SEED = 42       # random seed for reproducibility
TRAIN_SPLIT = 0.8     # training size (in fraction)

# Models
# list of supported models and their dataprep functions
ARGS_DICT = {
        # command line arg  |        model        |  dataprep function
        'linear':            [LinearRegression(),  dataprep_linear],
        'ridge':             [Ridge(),             dataprep_linear],
        'lasso':             [Lasso(),             dataprep_linear],
        'xgboost':           [XGBRegressor(),      dataprep_xgboost],
    }

# Evaluation metrics
METRICS = {
    'r2': r2_score,
    'mae':  mean_absolute_error
}
