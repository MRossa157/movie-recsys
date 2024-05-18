from abc import ABC, abstractmethod

import pandas as pd

from src.constants import MOVIE_PATH


class BaseRecommender(ABC):
    def __init__(self, model_path: str, ) -> None:
        ...

    @abstractmethod
    def get_recommendation(self, ):
        ...

    @abstractmethod
    def get_recommend_for_new_user(self):
        ...