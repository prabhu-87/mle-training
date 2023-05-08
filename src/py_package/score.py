import os
import tarfile
import argparse
import numpy as np
import pandas as pd
from scipy.stats import randint
from six.moves import urllib
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.tree import DecisionTreeRegressor
import logging
import pickle

from ingest_data import load_housing_data

root_directory = str(os.path.dirname(os.path.abspath(__file__)))
os.chdir(root_directory + "/../..")

processed_data_path = os.path.join("datasets", "processed_data_path")
parser = argparse.ArgumentParser("scoring the model")

parser.add_argument(
    "--train_test_path",
    metavar="train_test_path",
    type=str,
    help="Give input to save data",
    default=processed_data_path,
)

parser.add_argument(
    "--model_path",
    metavar="model_path",
    type=str,
    help="give path for model",
    default="models",
)

parser.add_argument(
    "--log_path",
    metavar="log_path",
    type=str,
    default=None,
    help="Enter log loc (default: %(default)s)",
)

parser.add_argument(
    "--log_level",
    metavar="log_level",
    help="Configure the logging level",
    default=logging.DEBUG,
)

parser.add_argument(
    "--no_console_log",
    help="Enter you console action (default:False)",
    action="store_false",
)

args, unknown = parser.parse_known_args()
log_path = args.log_path
train_test_path = args.train_test_path
log_level = args.log_level
model_path = args.model_path

logger = logging.getLogger(__name__)
formatter = logging.Formatter("%(asctime)s:%(name)s:%(message)s")
logger.setLevel(log_level)

if args.log_path:
    LOG_PATH = os.path.join(log_path)
    os.makedirs(LOG_PATH, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_path, "score.log"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
else:
    pass

if args.no_console_log:
    console_log = True

else:
    console_log = False
    logger.propagate = False

if console_log == True:
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

else:
    pass


def score():

    """
    this function checks score for each model.

    Parameters:
        - None

    **Returs:** None

    """
    pipeline = pickle.load(open(os.path.join("artifacts", "pipeline.pkl"), "rb"))

    x_test = pd.read_csv(os.path.join(train_test_path, "start_test_set.csv"))
    x_test.drop(columns=["median_house_value"], axis=1, inplace=True)
    x_test_prepared = pipeline.fit_transform(x_test)
    y_test = pd.read_csv(os.path.join(train_test_path, "y_test.csv"))
    y_test.drop(columns=y_test.columns[0], axis=1, inplace=True)

    strat_train_set = pd.read_csv(os.path.join(train_test_path, "start_train_set.csv"))
    housing = strat_train_set.drop(
        "median_house_value", axis=1
    )  # drop labels for training set
    housing_prepared = pipeline.fit_transform(housing)
    housing_labels = pd.read_csv(os.path.join(train_test_path, "housing_labels.csv"))
    housing_labels.drop(columns=housing_labels.columns[0], axis=1, inplace=True)

    print(housing_prepared)
    print("-" * 50)
    print(x_test_prepared)
    # load the model from disk
    LR_Path = os.path.join(model_path, "LinearRegression.sav")
    lin_reg = pickle.load(open(LR_Path, "rb"))
    housing_predictions = lin_reg.predict(housing_prepared)
    lin_mse = mean_squared_error(housing_labels, housing_predictions)
    lin_rmse = np.sqrt(lin_mse)
    lin_mae = mean_absolute_error(housing_labels, housing_predictions)

    logger.debug("Linear Regression")
    logger.debug("MSE = {}".format(lin_mse))
    logger.debug("RMSE = {}".format(lin_rmse))
    logger.debug("MAE = {}".format(lin_mae))

    # load the model from disk

    DT_Path = os.path.join(model_path, "DecisionTreeRegressor.sav")
    tree_reg = pickle.load(open(DT_Path, "rb"))
    housing_predictions = tree_reg.predict(housing_prepared)
    tree_mse = mean_squared_error(housing_labels, housing_predictions)
    tree_rmse = np.sqrt(tree_mse)
    tree_mae = mean_absolute_error(housing_labels, housing_predictions)

    logger.debug("Decision Tree Regressor")
    logger.debug("MSE = {}".format(tree_mse))
    logger.debug("RMSE = {}".format(tree_rmse))

    RFRSCV_Path = os.path.join(model_path, "RandomForestRegressorRSCV.sav")
    rnd_search = pickle.load(open(RFRSCV_Path, "rb"))
    cvres = rnd_search.cv_results_

    logger.debug("Random Forest Regressor RSCV")
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        logger.debug("MSE = {}, params = {}".format(np.sqrt(-mean_score), params))

    GSCV_Path = os.path.join(model_path, "RandomForestRegressorGSCV.sav")
    gd_search = pickle.load(open(GSCV_Path, "rb"))
    cvres = gd_search.cv_results_

    logger.debug("Random Forest Regressor GSCV")
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        logger.debug("MSE = {}, params = {}".format(np.sqrt(-mean_score), params))

    FM_Path = os.path.join(model_path, "FinalModel.sav")
    final_model = pickle.load(open(FM_Path, "rb"))
    final_predictions = final_model.predict(x_test_prepared)
    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)

    logger.debug("Final Model")
    logger.debug("MSE = {}".format(final_mse))
    logger.debug("RMSE = {}".format(final_rmse))


if __name__ == "__main__":
    score()
