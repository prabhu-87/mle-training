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
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     StratifiedShuffleSplit, train_test_split)
from sklearn.tree import DecisionTreeRegressor
import logging
import pickle



DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
processed_data_path = os.path.join("datasets", "processed_data_path")


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    """
    This fetch_housing_data function will fetch the data from the given url
    and creates data's tarfile to given path.

    Parameters:
        - **housing_url** (list) - url of data
        - **housing_path** (str) - path to store data

    **Returs:** None


    """

    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

def load_housing_data(housing_path=HOUSING_PATH):
    """
    this function loads the data from given path.

    Parameters:
        - **housing_path** (str): path of saved data

    **Returs:** csv file of data

    """
    fetch_housing_data()
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

parser = argparse.ArgumentParser("Create training, validation datasets")
parser.add_argument("--train_test_path",
                    metavar="train_test_path",
                    type=str,
                    help="enter location to save data",
                    default=processed_data_path,
                    )

parser.add_argument("--log_level",
                    metavar="log_level",
                    help="enter log level (default: Debug)",
                    default=logging.DEBUG,
)

parser.add_argument(
    "--log_path",
    metavar="log_path",
    type=str,
    default=None,
    help="give input for log loc (default: %(default)s)",
)

parser.add_argument(
    "--no-console-log",
    help="do you want to print on console (default:print on console)",
    action="store_false"
)

args, unknown = parser.parse_known_args()
log_path = args.log_path
train_test_path = args.train_test_path
log_level = args.log_level

logger = logging.getLogger(__name__)
formatter = logging.Formatter("%(asctime)s:%(name)s:%(message)s")
logger.setLevel(log_level)

if args.log_path:
    LOG_PATH = os.path.join(log_path)
    os.makedirs(LOG_PATH, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_path, "ingest_data.log"))
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

def generate_file(train_test_path):
    """
    this function creates train and test data and store them in the given path.

    Parameters:
        - **train_test_path** (str): give path to save data

    **Returs:** None

    """

    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1,2,3,4,5],
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
    logger.info(train_test_path)
    os.makedirs(train_test_path, exist_ok=True)
    train_set.to_csv(os.path.join(train_test_path, "train_raw.csv"))
    test_set.to_csv(os.path.join(train_test_path, "test_raw.csv"))
    strat_train_set.to_csv(os.path.join(train_test_path, "start_train_set.csv"))
    strat_test_set.to_csv(os.path.join(train_test_path, "start_test_set.csv"))


if __name__=="__main__":

    fetch_housing_data(HOUSING_URL, HOUSING_PATH)
    housing = load_housing_data(HOUSING_PATH)
    generate_file(train_test_path)
