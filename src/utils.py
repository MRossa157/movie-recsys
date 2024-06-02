import pandas as pd


#! depricated
class MovieEncoder:
    def __init__(self, movie_csv_path) -> None:
        self.movie_data = pd.read_csv(movie_csv_path)

        self.id_to_title = self.movie_data.set_index("movieId")["title"].to_dict()
        self.title_to_id = {v: k for k, v in self.id_to_title.items()}

    def to_idx(self, title):
        return self.title_to_id.get(title, None)

    def to_title(self, idx):
        return self.id_to_title.get(idx, None)

    def num_products(self):
        return self.movie_data.shape[0]


class MovieMapper:
    def __init__(self, movie_data_path):
        # Загружаем данные о фильмах
        self.movie_data = pd.read_csv(movie_data_path)
        # Создаем словари для быстрого поиска
        self.movie_id_to_title = pd.Series(
            self.movie_data.title.values, index=self.movie_data.movieId
        ).to_dict()
        self.title_to_id = pd.Series(
            self.movie_data.movieId.values, index=self.movie_data.title
        ).to_dict()

    def movieid_to_title(self, movieid: int) -> str | None:
        # Возвращаем название фильма по его ID
        return self.movie_id_to_title.get(movieid, None)

    def title_to_movieid(self, title: str) -> int | None:
        # Возвращаем ID фильма по его названию
        return self.title_to_id.get(title, None)


def average_precision(actual, recommended, k=6):
    ap_sum = 0
    hits = 0
    for i in range(k):
        product_id = recommended[i] if i < len(recommended) else None
        if product_id is not None and product_id in actual:
            hits += 1
            ap_sum += hits / (i + 1)
    return ap_sum / min(k, len(actual))


def mean_average_precision(actual_dict, recommended_dict, k=6):
    total_ap = 0
    users_count = 0

    for user_id, actual in actual_dict.items():
        recommended = recommended_dict.get(user_id, [])

        # Считаем AP только для тех пользователей, у которых есть реальные данные
        if actual:
            actual_set = set(actual)
            ap = average_precision(actual_set, recommended, k=k)
            total_ap += ap
            users_count += 1

    # Возвращаем среднее значение AP по всем пользователям, у которых были актуальные данные
    return total_ap / users_count if users_count > 0 else 0


def normalized_average_precision(actual_dict, recommended_dict, k=6):
    total_nap = 0
    users_count = len(actual_dict)
    for user_id, actual in actual_dict.items():
        recommended = recommended_dict.get(user_id, [])

        actual_set = set(actual)
        if len(actual_set) == 0:
            continue

        ap = average_precision(actual_set, recommended, k=k)
        ap_ideal = average_precision(actual_set, list(actual_set)[:k], k=k)

        total_nap += ap / ap_ideal if ap_ideal != 0 else 0

    return total_nap / users_count


def train_test_split(ratings: pd.DataFrame):
    ratings = ratings.copy()
    ratings["rank_latest"] = ratings.groupby(["userId"])["timestamp"].rank(
        method="first", ascending=False
    )

    train_ratings = ratings[ratings["rank_latest"] != 1]
    test_ratings = ratings[ratings["rank_latest"] == 1]

    # дропаем колонки которые нам уже не нужны (timestamp)
    train_ratings = train_ratings[["userId", "movieId", "rating"]]
    test_ratings = test_ratings[["userId", "movieId", "rating"]]
    return train_ratings, test_ratings
