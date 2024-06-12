import logging
import os

import implicit
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

import constants

BEST_PARAMS = {'factors': 60, 'iterations': 15, 'regularization': 0.01}


def train(
    csr_train, factors=200, iterations=15, regularization=0.01, show_progress=True
):
    model = implicit.als.AlternatingLeastSquares(
        factors=factors,
        iterations=iterations,
        regularization=regularization,
        random_state=42,
    )
    model.fit(csr_train, show_progress=show_progress)
    return model


def to_user_item_coo(df: pd.DataFrame):
    """Turn a dataframe with transactions into a COO sparse items x users matrix"""
    row = df["mapped_user_id"].values
    col = df["mapped_movie_id"].values
    data = np.ones(df.shape[0])
    coo = coo_matrix((data, (row, col)))
    return coo


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logging.info("Считываем данные")
    ratings = pd.read_csv(constants.RATINGS_PATH)
    movies = pd.read_csv(constants.MOVIE_PATH)
    logging.info("Предобрабатываем данные")
    ALL_USERS = ratings["userId"].unique().tolist()
    ALL_ITEMS = movies["movieId"].unique().tolist()

    user_ids = dict(list(enumerate(ALL_USERS)))
    item_ids = dict(list(enumerate(ALL_ITEMS)))

    user_map = {u: uidx for uidx, u in user_ids.items()}
    item_map = {i: iidx for iidx, i in item_ids.items()}

    ratings["mapped_user_id"] = ratings["userId"].map(user_map)
    ratings["mapped_movie_id"] = ratings["movieId"].map(item_map)

    coo_train = to_user_item_coo(ratings)
    csr_train = coo_train.tocsr()

    logging.info("Запускаем обучение")
    model = train(csr_train, **BEST_PARAMS)

    logging.info(f"Сохраняем веса в папку {constants.WEIGHTS_PATH}")
    if not os.path.exists(constants.WEIGHTS_PATH):
        os.makedirs(constants.WEIGHTS_PATH)

    model.save(rf"{constants.WEIGHTS_PATH}/als.npz")
