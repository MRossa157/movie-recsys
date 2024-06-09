import enum
import warnings

import constants
import implicit
import numpy as np
import pandas as pd
from implicit.evaluation import mean_average_precision_at_k
from scipy.sparse import coo_matrix
from utils import train_test_split

warnings.filterwarnings("ignore")


class Device(enum.Enum):
    CPU = "cpu"
    GPU = "gpu"


class GridSearcher:
    def __init__(
        self,
        ratings: pd.DataFrame,
        movies: pd.DataFrame,
        device: Device = Device.CPU,
        show_progress=True,
    ) -> None:
        self.device = device
        self.ratings = ratings.copy()
        self.movies = movies.copy()
        self.show_progress = show_progress

        # In train propouses we will use only 30% of all ratings dataset
        rand_userIds = np.random.choice(
            self.ratings["userId"].unique(),
            size=int(len(self.ratings["userId"].unique()) * 0.3),
            replace=False,
        )

        self.ratings = self.ratings.loc[self.ratings["userId"].isin(rand_userIds)]

        ALL_USERS = self.ratings["userId"].unique().tolist()
        ALL_ITEMS = self.movies["movieId"].unique().tolist()

        user_ids = dict(list(enumerate(ALL_USERS)))
        item_ids = dict(list(enumerate(ALL_ITEMS)))

        user_map = {u: uidx for uidx, u in user_ids.items()}
        item_map = {i: iidx for iidx, i in item_ids.items()}

        self.ratings["mapped_user_id"] = self.ratings["userId"].map(user_map)
        self.ratings["mapped_movie_id"] = self.ratings["movieId"].map(item_map)

    def run(
        self,
    ):
        matrices = self.get_val_matrices(self.ratings)

        # Grid Search
        regularization_params = [0, 0.1, 0.01]
        iter_params = [3, 12, 14, 15, 20]
        factors_params = [40, 50, 60, 100, 200, 500, 1000]

        best_map = 0
        for regularization in regularization_params:
            for iterations in iter_params:
                for factors in factors_params:
                    map = self.validate(
                        matrices,
                        factors,
                        iterations,
                        regularization,
                    )
                    if map > best_map:
                        best_map = map
                        best_params = {
                            "factors": factors,
                            "iterations": iterations,
                            "regularization": regularization,
                        }
                        print(f"Best MAP@30 found. Updating: {best_params}")

    def to_user_item_coo(self, df: pd.DataFrame):
        """Turn a dataframe with transactions into a COO sparse items x users matrix"""
        row = df["mapped_user_id"].values
        col = df["mapped_movie_id"].values
        data = np.ones(df.shape[0])
        coo = coo_matrix((data, (row, col)))
        return coo

    def get_val_matrices(self, df: pd.DataFrame):
        """
        Returns a dictionary with the following keys:
                csr_train: training data in CSR sparse format and as (users x items)
                csr_val:  validation data in CSR sparse format and as (users x items)
        """
        df_train, df_test = train_test_split(df)

        coo_train = self.to_user_item_coo(df_train)
        coo_test = self.to_user_item_coo(df_test)

        csr_train = coo_train.tocsr()
        csr_test = coo_test.tocsr()

        return {"csr_train": csr_train, "csr_test": csr_test}

    def validate(
        self,
        matrices: dict,
        factors=200,
        iterations=20,
        regularization=0.01,
    ):
        """Train an ALS model with <<factors>> (embeddings dimension)
        for <<iterations>> over matrices and validate with Mean Average Precision
        """
        csr_train, csr_test = matrices["csr_train"], matrices["csr_test"]

        if self.device == Device.CPU:
            model = implicit.cpu.als.AlternatingLeastSquares(
                factors=factors, iterations=iterations, regularization=regularization
            )
        else:
            model = implicit.gpu.als.AlternatingLeastSquares(
                factors=factors, iterations=iterations, regularization=regularization
            )
        model.fit(csr_train, show_progress=self.show_progress)

        metric_map = mean_average_precision_at_k(
            model, csr_train, csr_test, K=6, show_progress=self.show_progress
        )
        print(
            f"Factors: {factors:>3} - Iterations: {iterations:>2} - Regularization: {regularization:4.3f} ==> MAP@6: {metric_map:6.5f}"
        )
        return metric_map


if __name__ == "__main__":
    ratings = pd.read_csv(constants.RATINGS_PATH)
    movies = pd.read_csv(constants.MOVIE_PATH)

    searcher = GridSearcher(ratings=ratings, movies=movies)
    searcher.run()
