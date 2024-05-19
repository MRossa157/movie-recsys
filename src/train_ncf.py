import logging
import re

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader, Dataset

import constants
from utils import train_test_split


class MovieLensTrainDataset(Dataset):
    """MovieLens PyTorch Dataset for Training

    Args:
        ratings (pd.DataFrame): Dataframe containing the movie ratings
        movies (pd.DataFrame): Dataframe containing information on all the movies

    """

    def __init__(self, ratings: pd.DataFrame, movies: pd.DataFrame):
        self.ratings = ratings
        self.users, self.items, self.genre, self.release_date, self.labels = self.get_dataset(ratings, movies)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.genre[idx], self.release_date[idx], self.labels[idx]

    @staticmethod
    def get_release_date_from_title(title: str) -> int | None:
        match = re.search(r'\((\d{4})\)', title)
        if match:
            return int(match.group(1))
        return None

    def get_dataset(self, ratings: pd.DataFrame, movies: pd.DataFrame):
        # Work with copies of the ratings and movies dataframes
        ratings = ratings.copy()
        movies = movies.copy()

        # Extract release dates from titles
        movies['release_date'] = movies['title'].apply(self.get_release_date_from_title)
        movies['release_date'] = movies['release_date'].fillna(0)
        movies['release_date'] = movies['release_date'].astype(int)

        # One-Hot Encoding for genres
        movies['genres'] = movies['genres'].apply(lambda x: x.split('|'))
        mlb = MultiLabelBinarizer()
        genres_encoded = mlb.fit_transform(movies['genres'])
        genres_df = pd.DataFrame(genres_encoded, columns=mlb.classes_, index=movies['movieId'])

        # Merge genres and release_date with ratings
        movies = movies.join(genres_df, on='movieId')
        merged_df = ratings.merge(movies, on='movieId')

        # Extracting final tensors
        users = torch.tensor(merged_df['userId'].values)
        items = torch.tensor(merged_df['movieId'].values)
        genres = torch.tensor(merged_df[mlb.classes_].values)
        release_dates = torch.tensor(merged_df['release_date'].values)
        labels = torch.tensor(merged_df['rating'].values)

        return users, items, genres, release_dates, labels

    def get_num_users(self) -> int:
        return self.ratings['userId'].max()

    def get_num_items(self) -> int:
        return self.ratings['movieId'].max()

    def get_num_genres(self) -> int:
        return self.genre.shape[1]

    def get_max_release_date(self) -> int:
        return self.release_date.numpy().max()

    def get_min_release_date(self) -> int:
        return self.release_date.numpy().min()

class NCF(pl.LightningModule):
    """ Neural Collaborative Filtering (NCF)

        Args:
            dataset (MovieLensTrainDataset): Dataset for training
    """

    def __init__(self, dataset: MovieLensTrainDataset):
        super().__init__()
        self.BATCH_SIZE = 512
        self.dataset = dataset

        self.user_embedding = nn.Embedding(num_embeddings=dataset.get_num_users() + 1, embedding_dim=8)
        self.item_embedding = nn.Embedding(num_embeddings=dataset.get_num_items() + 1, embedding_dim=8)
        self.genre_embedding = nn.Embedding(num_embeddings=dataset.get_num_genres() + 1, embedding_dim=8)

        # min_release_date = dataset.get_min_release_date()
        # max_release_date = dataset.get_max_release_date()
        # num_date_embeddings = max_release_date - min_release_date + 2  # +2 to account for the -1 fill value and inclusive range

        self.date_embedding = nn.Embedding(num_embeddings=dataset.get_max_release_date() + 1, embedding_dim=8)

        self.genre_fc = nn.Linear(in_features=dataset.get_num_genres() * 8, out_features=8)

        input_size = 8 + 8 + 8 + 8  # user_embedded + item_embedded + genre_embedded + date_embedded
        self.fc1 = nn.Linear(in_features=input_size, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.output = nn.Linear(in_features=32, out_features=1)

    def forward(self, user_input, item_input, genre_input, date_input):
        # Pass through embedding layers
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)
        genre_embedded = self.genre_embedding(genre_input)
        # Flatten the genre embeddings and pass through linear layer
        genre_embedded = genre_embedded.view(genre_embedded.size(0), -1)
        genre_embedded = self.genre_fc(genre_embedded)

        # Shift date input to start from 0 for embedding lookup
        # min_release_date = self.dataset.get_min_release_date()
        # date_input = date_input - min_release_date + 1
        date_embedded = self.date_embedding(date_input)

        # Item embedding = movie_id + movie_genre + movie_release_date
        item_embedded = torch.cat([item_embedded, genre_embedded, date_embedded], dim=-1)

        # Concat the two embedding layers
        vector = torch.cat([user_embedded, item_embedded], dim=-1)

        # Pass through dense layers
        vector = nn.ReLU()(self.fc1(vector))
        vector = nn.ReLU()(self.fc2(vector))

        # Output layer with clamp to ensure rating range
        pred = torch.clamp(self.output(vector), min=1.0, max=5.0)

        return pred

    def training_step(self, batch, batch_idx):
        user_input, item_input, genre_input, date_input, labels = batch
        predicted_labels = self(user_input, item_input, genre_input, date_input)
        loss = nn.MSELoss()(predicted_labels, labels.view(-1, 1).float())
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=512, num_workers=5, persistent_workers=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info('Считываем данные')
    ratings = pd.read_csv(constants.RATINGS_PATH, parse_dates=['timestamp'], nrows=5962811)
    movies = pd.read_csv(constants.MOVIE_PATH)

    # TAKE 30% HERE
    logging.info('Предобрабатываем данные')
    # rand_userIds = np.random.choice(ratings['userId'].unique(),
    #                             size=int(len(ratings['userId'].unique())*0.3),
    #                             replace=False)

    # ratings = ratings.loc[ratings['userId'].isin(rand_userIds)]

    train_ratings, test_ratings = train_test_split(ratings)

    # Init NCF model
    logging.info('Инициализируем модель')
    train_dataset = MovieLensTrainDataset(train_ratings, movies)
    model = NCF(train_dataset)

    checkpoint_callback = ModelCheckpoint(dirpath=r'srt/weights/',
                                          filename='{epoch}-{train_loss:.2f}',
                                          monitor="train_loss")
    trainer = pl.Trainer(max_epochs=5,
                        devices="auto", accelerator="auto",
                        fast_dev_run=False,
                        logger=False,
                        callbacks=[checkpoint_callback])

    logging.info('Запускаем обучение')
    trainer.fit(model)
    trainer.save_checkpoint(r"src/weights/NCF_FINAL_epoch.ckpt")
