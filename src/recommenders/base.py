from abc import ABC, abstractmethod


class BaseRecommender(ABC):
    def __init__(
        self,
        model_path: str,
    ) -> None: ...

    @abstractmethod
    def get_recommendation(
        self,
    ):
        raise NotImplementedError()

    @abstractmethod
    def get_recommend_for_new_user(self):
        raise NotImplementedError()
