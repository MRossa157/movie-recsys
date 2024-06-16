import json

import numpy as np
import pandas as pd
from base import BaseRecommender
from implicit.cpu.als import AlternatingLeastSquares
from scipy.sparse import coo_matrix, csr_matrix

import src.constants as constants
from src.utils import MovieMapper


class ALSRecommender(BaseRecommender):
    def __init__(self, model_path, ratings, movies) -> None:
        """
        model_path (str): Path to .npz checkpoint file from lib 'implicit' ALS model

        ratings (pd.DataFrame): Dataframe containing the movie ratings that it was trained on
        movies (pd.DataFrame): Dataframe containing the data movies
        """
        if not model_path.endswith(".npz"):
            raise ValueError("Путь к модели должен содержать файл с расширением .npz")
        self.model = AlternatingLeastSquares.load(model_path)
        self.ratings = ratings
        self.movies = movies

        self.__create_mappings(ratings, movies)

        ready_ratings = self.__prepare_ratings(self.ratings)
        self.user_items = ALSRecommender.to_user_item_coo(ready_ratings).tocsr()

    def __create_mappings(self, ratings, movies):
        self.ALL_USERS = ratings["userId"].unique().tolist()
        self.ALL_ITEMS = movies["movieId"].unique().tolist()

        self.user_ids = dict(list(enumerate(self.ALL_USERS)))
        self.item_ids = dict(list(enumerate(self.ALL_ITEMS)))

        self.user_map = {u: uidx for uidx, u in self.user_ids.items()}
        self.item_map = {i: iidx for iidx, i in self.item_ids.items()}

    def get_mapped_user(self, user_id):
        return self.user_map.get(user_id)

    def get_mapped_item(self, item_id):
        return self.item_map.get(item_id)

    def __prepare_ratings(self, ratings):
        ratings = ratings.copy()
        ratings["mapped_user_id"] = ratings["userId"].map(self.user_map)
        ratings["mapped_movie_id"] = ratings["movieId"].map(self.item_map)
        return ratings

    @staticmethod
    def to_user_item_coo(df: pd.DataFrame):
        """Turn a dataframe with transactions into a COO sparse items x users matrix"""
        row = df["mapped_user_id"].values
        col = df["mapped_movie_id"].values
        data = np.ones(df.shape[0])
        coo = coo_matrix((data, (row, col)))
        return coo

    @staticmethod
    def print_recommendations(movie_mapper: MovieMapper, item_ids, scores):
        for movie_id, score in zip(item_ids, scores):
            print(
                f"Movie ID: {movie_id}, Movie Title: {movie_mapper.movieid_to_title(movie_id)}, Score: {score}"
            )

    def get_recommendation(
        self,
        userid,
        N=12,
        filter_already_liked_items=False,
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
        customer_id = self.get_mapped_user(userid)

        ids, scores = self.model.recommend(
            customer_id,
            self.user_items[customer_id],
            N=N,
            filter_already_liked_items=filter_already_liked_items,
            filter_items=filter_items,
            recalculate_user=recalculate_user,
            items=items,
        )

        mapped_ids = [self.item_ids.get(item_id) for item_id in ids]

        return mapped_ids, scores

    def get_recommendation_for_new_user(self, ratings, n_recommendations=6):
        """
        Get recommendations for a new user.

        Parameters
        ratings (dict) - Dictionary with new user's ratings, where keys are movieId and values are ratings (float).
        n_recommendations (int, optional) - Number of recommendations to return

        Returns
        Tuple of (itemids, scores) arrays. For a single user, these arrays will be 1-dimensional with N items.
        """

        num_items = self.user_items.shape[1]

        temp_user_items = csr_matrix((1, num_items))

        for movie_id, _ in ratings.items():
            mapped_movie_id = self.get_mapped_item(movie_id)
            # Since this is an implicit method, we simply mark the element that the user interacted with
            temp_user_items[0, mapped_movie_id] = 1

        # userid is not important because recalculate_user=True
        ids, scores = self.model.recommend(
            0, temp_user_items, N=n_recommendations, recalculate_user=True
        )

        mapped_ids = [self.item_ids.get(item_id) for item_id in ids]

        return mapped_ids, scores


if __name__ == "__main__":
    from_custom_ratings = False
    model_path = rf"{constants.WEIGHTS_PATH}/als.npz"
    ratings = pd.read_csv(constants.RATINGS_PATH)
    movies = pd.read_csv(constants.MOVIE_PATH)

    recommender = ALSRecommender(model_path, ratings, movies)

    if from_custom_ratings:
        with open(
            r"custom_user_ratings/egor_ratings.json", "r", encoding="utf-8"
        ) as file:
            new_user_ratings = json.load(file)
            new_user_ratings = {
                int(movieid): float(rating)
                for movieid, rating in new_user_ratings.items()
            }
    else:
        # Star Wars Fan
        new_user_ratings = {
            5378: 5,
            33493: 5,
            61160: 5,
            79006: 4,
            100089: 5,
            109713: 5,
            260: 5,
            1196: 5,
        }

    recommendations = recommender.get_recommendation_for_new_user(
        new_user_ratings, n_recommendations=6
    )

    movie_mapper = MovieMapper(constants.MOVIE_PATH)

    ALSRecommender.print_recommendations(
        movie_mapper, recommendations[0], recommendations[1]
    )
