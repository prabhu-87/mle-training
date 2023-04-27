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
import matplotlib
from sklearn.tree import DecisionTreeRegressor
import logging
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from utility import CombinedAttributesAdder
from ingest_data import load_housing_data


root_directory = str(os.path.dirname(os.path.abspath(__file__)))
os.chdir(root_directory + "/../..")
housing_path = os.path.join("datasets", "housing_raw")
processed_data_path = os.path.join("datasets", "processed_data_path")


def income_cat_proportions(data):
    """
    this function creates proportion of income categories.

    Parameters:
        - **data** (pd.DataFrame): data from which we want to calculate proportion

    **Returs:** array having proportions of categories

    """
    return data["income_cat"].value_counts() / len(data)


parser = argparse.ArgumentParser(description="training the model")

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
    "--log_level",
    metavar="log_level",
    help="Configure the logging level",
    default=logging.DEBUG,
)

parser.add_argument(
    "--log_path",
    metavar="log_path",
    type=str,
    default=None,
    help="Enter log loc (default: %(default)s)",
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
    file_handler = logging.FileHandler(os.path.join(log_path, "train.log"))
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
# column index
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6


new_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("attribs_adder", CombinedAttributesAdder()),
        ("std_scaler", StandardScaler()),
    ]
)

Artifact_Path = os.path.join("artifacts")
os.makedirs(Artifact_Path, exist_ok=True)
pipeline_dir = os.path.join(Artifact_Path, "numerical_pipeline.pkl")
pickle.dump(new_pipeline, open(pipeline_dir, "wb"))


def train(model_path):

    """
    this function trains all the model and pickle them in given path.

    Parameters:
        - **model_path** (str): give path to save all model pickles

    **Returs:** None

    """

    housing = load_housing_data()
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

    compare_props = pd.DataFrame(
        {
            "Overall": income_cat_proportions(housing),
            "Stratified": income_cat_proportions(strat_test_set),
            "Random": income_cat_proportions(test_set),
        }
    ).sort_index()
    compare_props["Rand. %error"] = (
        100 * compare_props["Random"] / compare_props["Overall"] - 100
    )
    compare_props["Strat. %error"] = (
        100 * compare_props["Stratified"] / compare_props["Overall"] - 100
    )

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    housing = strat_train_set.copy()
    housing.plot(kind="scatter", x="longitude", y="latitude")
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

    corr_matrix = housing.corr()
    corr_matrix["median_house_value"].sort_values(ascending=False)

    housing = strat_train_set.drop(
        "median_house_value", axis=1
    )  # drop labels for training set
    housing_labels = strat_train_set["median_house_value"].copy()
    housing_num = housing.drop("ocean_proximity", axis=1)
    num_attributes = list(housing_num)

    cat_attributes = ["ocean_proximity"]
    housing_cat = housing[["ocean_proximity"]]
    pipeline_PATH = os.path.join("artifacts", "numerical_pipeline.pkl")
    num_pipeline_loaded = pickle.load(open(pipeline_PATH, "rb"))

    full_pipeline = ColumnTransformer(
        [
            ("num", new_pipeline, num_attributes),
            ("cat", OneHotEncoder(), cat_attributes),
        ]
    )

    Artifact_Path = os.path.join("artifacts")
    os.makedirs(Artifact_Path, exist_ok=True)
    pipeline_dir = os.path.join(Artifact_Path, "pipeline.pkl")
    pickle.dump(full_pipeline, open(pipeline_dir, "wb"))
    pipeline_loaded = pickle.load(open(pipeline_PATH, "rb"))

    housing_prepared = full_pipeline.fit_transform(housing)

    MODEL_PATH = os.path.join(model_path)
    os.makedirs(MODEL_PATH, exist_ok=True)

    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)
    filename = "LinearRegression.sav"
    LR_DIR = os.path.join(model_path, filename)
    pickle.dump(lin_reg, open(LR_DIR, "wb"))

    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg.fit(housing_prepared, housing_labels)
    filenamee = "DecisionTreeRegressor.sav"
    RFR_DIR = os.path.join(model_path, filenamee)
    pickle.dump(tree_reg, open(RFR_DIR, "wb"))

    param_distribs = {
        "n_estimators": randint(low=1, high=200),
        "max_features": randint(low=1, high=8),
    }

    forest_reg = RandomForestRegressor(random_state=42)
    rnd_search = RandomizedSearchCV(
        forest_reg,
        param_distributions=param_distribs,
        n_iter=10,
        cv=5,
        scoring="neg_mean_squared_error",
        random_state=42,
    )
    rnd_search.fit(housing_prepared, housing_labels)
    filename = "RandomForestRegressorRSCV.sav"
    RFR_DIR = os.path.join(model_path, filename)
    pickle.dump(rnd_search, open(RFR_DIR, "wb"))

    param_grid = [
        # try 12 (3×4) combinations of hyperparameters
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        # then try 6 (2×3) combinations with bootstrap set as False
        {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
    ]

    forest_reg = RandomForestRegressor(random_state=42)
    # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
    grid_search = GridSearchCV(
        forest_reg,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        return_train_score=True,
    )

    grid_search.fit(housing_prepared, housing_labels)
    filename = "RandomForestRegressorGSCV.sav"
    RFR_DIR2 = os.path.join(model_path, filename)
    pickle.dump(rnd_search, open(RFR_DIR2, "wb"))

    extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
    feature_importances = grid_search.best_estimator_.feature_importances_
    cat_encoder = full_pipeline.named_transformers_["cat"]
    cat_one_hot_attribs = list(cat_encoder.categories_[0])
    attributes = num_attributes + extra_attribs + cat_one_hot_attribs
    sorted(zip(feature_importances, attributes), reverse=True)

    final_model = grid_search.best_estimator_

    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()

    X_test_num = X_test.drop("ocean_proximity", axis=1)
    X_test_num_attributes = list(X_test_num)
    X_test_cat_attributes = ["ocean_proximity"]
    X_test_cat = X_test[["ocean_proximity"]]
    X_test_prepared = full_pipeline.fit_transform(X_test)

    # X_test_prepared.to_csv(os.path.join(train_test_path, "X_test_prepared.csv"))
    y_test.to_csv(os.path.join(train_test_path, "y_test.csv"))
    # housing_prepared.to_csv(os.path.join(train_test_path, "housing_prepared.csv"))
    housing_labels.to_csv(os.path.join(train_test_path, "housing_labels.csv"))

    filenamee = "FinalModel.sav"
    FINAL_DIR = os.path.join(model_path, filenamee)
    pickle.dump(final_model, open(FINAL_DIR, "wb"))


if __name__ == "__main__":
    train(model_path)
