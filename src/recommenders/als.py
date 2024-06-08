import json

import pandas as pd
from base import BaseRecommender
from implicit.cpu.als import AlternatingLeastSquares
from scipy.sparse import coo_matrix, csr_matrix

from src.constants import MOVIE_PATH, RATINGS_PATH, WEIGHTS_PATH
from src.utils import MovieMapper


class ALSRecommender(BaseRecommender):
    def __init__(self, model_path, user_items) -> None:
        """
        model_path (str) = Path to .npz checkpoint file from lib 'implicit' ALS model
        user_items (csr_matrix) – A sparse matrix of shape (users, number_items). This lets us look up the liked items and their weights for the user. This is used to filter out items that have already been liked from the output, and to also potentially recalculate the user representation. Each row in this sparse matrix corresponds to a row in the userid parameter: that is the first row in this matrix contains the liked items for the first user in the userid array.
        """
        if not model_path.endswith(".npz"):
            raise ValueError("Путь к модели должен содержать файл с расширением .npz")
        self.model = AlternatingLeastSquares.load(model_path)
        self.user_items = user_items

    def get_recommendation(
        self,
        userid,
        N=10,
        filter_already_liked_items=True,
        filter_items=None,
        recalculate_user=False,
        items=None,
    ) -> tuple:
        """
        Parameters
        userid (Union[int, array_like]) – The userid or array of userids to calculate recommendations for

        N (int, optional) – The number of results to return

        filter_already_liked_items (bool, optional) – When true, don’t return items present in the training set that were rated by the specified user.

        filter_items (array_like, optional) – List of extra item ids to filter out from the output

        recalculate_user (bool, optional) – When true, don’t rely on stored user embeddings and instead recalculate from the passed in user_items. This option isn’t supported by all models.

        items (array_like, optional) – Array of extra item ids. When set this will only rank the items in this array instead of ranking every item the model was fit for. This parameter cannot be used with filter_items

        Returns
        Tuple of (itemids, scores) arrays. For a single user these array will be 1-dimensional with N items.
        """
        return self.model.recommend(
            userid,
            self.user_items[userid],
            N=N,
            filter_already_liked_items=filter_already_liked_items,
            filter_items=filter_items,
            recalculate_user=recalculate_user,
            items=items,
        )

    def get_recommend_for_new_user(self, ratings, n_recommendations=6):
        """
        Получаем рекомендации для нового пользователя. Если recalculate=True, данные нового пользователя добавляются в основную матрицу.

        Parameters
        ratings (dict) – Словарь с оценками нового пользователя, где ключами являются movieId, а значениями - оценки (float)
        n_recommendations (int, optional) – Количество рекомендаций для возврата

        Returns
        Tuple of (itemids, scores) arrays. Для одного пользователя эти массивы будут 1-мерными с N элементами.
        """
        num_items = self.user_items.shape[1]

        temp_user_items = csr_matrix((1, num_items))
        for movie_id, rating in ratings.items():
            temp_user_items[0, movie_id] = rating
        # userid (в нашем случае 0) вообще не важен т.к. recalculate_user=True
        recommendations = self.model.recommend(
            0, temp_user_items, N=n_recommendations, recalculate_user=True
        )

        return recommendations

    @staticmethod
    def to_user_item_coo(df, shape):
        """Turn a dataframe with transactions into a COO sparse items x users matrix
        Parameters
        df (DataFrame) - Набор данных которые нужно переделать в COO матрицу
        shape (tuple) - Размерность матрицы (num_users, num_items)
        """
        row = df["userId"].values
        col = df["movieId"].values
        data = df["rating"].values
        coo = coo_matrix((data, (row, col)), shape=shape)
        return coo


if __name__ == "__main__":
    model_path = rf"{WEIGHTS_PATH}/als.npz"
    ratings = pd.read_csv(RATINGS_PATH)

    max_user_id = ratings["userId"].max()
    max_movie_id = ratings["movieId"].max()

    user_items = ALSRecommender.to_user_item_coo(
        ratings, (max_user_id + 1, max_movie_id + 1)
    ).tocsr()
    recommender = ALSRecommender(model_path, user_items)

    with open(r"custom_user_ratings\egor_ratings.json", "r", encoding="utf-8") as file:
        new_ratings = json.load(file)
        new_ratings = {
            int(movieid): float(rating) for movieid, rating in new_ratings.items()
        }

    # Star Wars Fan
    # new_ratings = {5378: 5, 33493: 5, 61160: 5, 79006: 4, 100089: 5, 109713: 5, 260: 5, 1196: 5}

    movie_mapper = MovieMapper(MOVIE_PATH)

    recommendations = recommender.get_recommend_for_new_user(
        new_ratings, n_recommendations=6
    )
    print("Recommendations for user ")
    item_ids, scores = recommendations
    for movie_id, score in zip(item_ids, scores):
        print(
            f"Movie ID: {movie_id}, Movie Title: {movie_mapper.movieid_to_title(movie_id)}, Score: {score}"
        )
