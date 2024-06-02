from abc import ABC, abstractmethod

from scipy.sparse import coo_matrix


class BaseRecommender(ABC):
    def __init__(
        self,
        model_path: str,
    ) -> None: ...

    @abstractmethod
    def get_recommendation(
        self,
    ): ...

    @abstractmethod
    def get_recommend_for_new_user(self): ...


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
