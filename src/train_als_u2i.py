import logging
import os

import implicit
import pandas as pd
from scipy.sparse import coo_matrix

from constants import RATINGS_PATH, WEIGHTS_PATH

BEST_PARAMS = {'factors': 100, 'iterations': 14, 'regularization': 0.1}

def train(csr_train, factors=200, iterations=15, regularization=0.01, show_progress=True):
    model = implicit.als.AlternatingLeastSquares(factors=factors,
                                                    iterations=iterations,
                                                    regularization=regularization,
                                                    random_state=42)
    model.fit(csr_train, show_progress=show_progress)
    return model

def to_user_item_coo(df: pd.DataFrame):
    """ Turn a dataframe with transactions into a COO sparse items x users matrix"""
    row = df['userId'].values
    col = df['movieId'].values
    data = df['rating'].values
    coo = coo_matrix((data, (row, col)))
    return coo

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info('Считываем данные')
    ratings = pd.read_csv(RATINGS_PATH)
    logging.info('Предобрабатываем данные')
    train_ratings = ratings[['userId', 'movieId', 'rating']]
    coo_train = to_user_item_coo(train_ratings)
    csr_train = coo_train.tocsr()

    logging.info('Запускаем обучение')
    model = train(csr_train, **BEST_PARAMS)

    logging.info(f'Сохраняем веса в папку {WEIGHTS_PATH}')
    if not os.path.exists(WEIGHTS_PATH):
        os.makedirs(WEIGHTS_PATH)

    model.save(rf'{WEIGHTS_PATH}/als.npz')