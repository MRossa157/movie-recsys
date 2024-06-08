import warnings

import implicit
import numpy as np
import pandas as pd
from implicit.evaluation import mean_average_precision_at_k
from pandas import DataFrame
from scipy.sparse import coo_matrix
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV

import constants
from utils import train_test_split

warnings.filterwarnings("ignore")


class ALSWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, factors=100, iterations=15, regularization=0.01):
        self.factors = factors
        self.iterations = iterations
        self.regularization = regularization
        self.model = None

    def fit(self, X, y=None):
        self.model = implicit.cpu.als.AlternatingLeastSquares(
            factors=self.factors,
            iterations=self.iterations,
            regularization=self.regularization,
        )

        self.model.fit(X)

        return self

    def predict(self, X):
        # For grid searching, this can just return a vector of zeros as actual prediction values are not used.
        return np.zeros(X.shape[0])


def map_scorer(estimator, X, y):
    csr_train, csr_val = X, y

    return mean_average_precision_at_k(estimator.model, csr_train, csr_val, K=6)


def to_user_item_coo(df: DataFrame):
    """Turn a dataframe with transactions into a COO sparse items x users matrix"""
    row = df["mapped_user_id"].values
    col = df["mapped_movie_id"].values
    data = np.ones(df.shape[0])
    coo = coo_matrix((data, (row, col)))
    return coo


def get_val_matrices(df: DataFrame):
    """
    Returns a dictionary with the following keys:
            csr_train: training data in CSR sparse format and as (users x items)
            csr_val:  validation data in CSR sparse format and as (users x items)
    """
    df_train, df_val = train_test_split(df)

    coo_train = to_user_item_coo(df_train)
    coo_val = to_user_item_coo(df_val)

    csr_train = coo_train.tocsr()
    csr_val = coo_val.tocsr()

    return {"csr_train": csr_train, "csr_val": csr_val}


def validate(
    matrices: dict, factors=200, iterations=20, regularization=0.01, show_progress=True
):
    """Train an ALS model with <<factors>> (embeddings dimension)
    for <<iterations>> over matrices and validate with Mean Average Precision
    """
    csr_train, csr_val = matrices["csr_train"], matrices["csr_val"]

    model = implicit.cpu.als.AlternatingLeastSquares(
        factors=factors, iterations=iterations, regularization=regularization
    )
    model.fit(csr_train, show_progress=show_progress)

    metric_map = mean_average_precision_at_k(
        model, csr_train, csr_val, K=6, show_progress=show_progress
    )
    print(
        f"Factors: {factors:>3} - Iterations: {iterations:>2} - Regularization: {regularization:4.3f} ==> MAP@6: {metric_map:6.5f}"
    )
    return metric_map


if __name__ == "__main__":
    ratings = pd.read_csv(constants.RATINGS_PATH)
    movies = pd.read_csv(constants.MOVIE_PATH)

    # In train propouses we will use only 30% of all ratings dataset
    rand_userIds = np.random.choice(
        ratings["userId"].unique(),
        size=int(len(ratings["userId"].unique()) * 0.3),
        replace=False,
    )

    ratings = ratings.loc[ratings["userId"].isin(rand_userIds)]
    print(
        "There are {} rows of data from {} users".format(
            len(ratings), len(rand_userIds)
        )
    )

    ALL_USERS = ratings["userId"].unique().tolist()
    ALL_ITEMS = movies["movieId"].unique().tolist()

    user_ids = dict(list(enumerate(ALL_USERS)))
    item_ids = dict(list(enumerate(ALL_ITEMS)))

    user_map = {u: uidx for uidx, u in user_ids.items()}
    item_map = {i: iidx for iidx, i in item_ids.items()}

    ratings["mapped_user_id"] = ratings["userId"].map(user_map)
    ratings["mapped_movie_id"] = ratings["movieId"].map(item_map)

    # train_ratings, test_ratings = train_test_split(ratings)

    matrices = get_val_matrices(ratings)
    csr_train, csr_val = matrices["csr_train"], matrices["csr_val"]

    # Set up GridSearchCV
    param_grid = {
        "factors": [40, 50, 60, 100, 200, 500, 1000],
        "iterations": [3, 12, 14, 15, 20],
        "regularization": [0, 0.1, 0.01],
    }

    model = ALSWrapper()
    grid = GridSearchCV(model, param_grid, cv=None, scoring=map_scorer, verbose=2)
    grid.fit(csr_train, csr_val)

    # Results
    best_params = grid.best_params_
    best_score = grid.best_score_
    print("Best Parameters:", best_params)
    print("Best Score:", best_score)
