import logging

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset

import constants
from utils import train_test_split


class MovieLensTrainDataset(Dataset):
    """MovieLens PyTorch Dataset for Training

    Args:
        ratings (pd.DataFrame): Dataframe containing the movie ratings
    """

    def __init__(self, ratings: pd.DataFrame):
        self.ratings = ratings.copy()
        self.users, self.items, self.labels = self.get_dataset(self.ratings)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]

    def get_dataset(self, ratings):
        # Перевод оценок
        ratings["rating"] *= 2  # умножаем на 2, чтобы перейти к 10-балльной шкале

        users = ratings["userId"].values
        items = ratings["movieId"].values
        labels = ratings["rating"].values

        return torch.tensor(users), torch.tensor(items), torch.tensor(labels)


class NCF(pl.LightningModule):
    """Neural Collaborative Filtering (NCF)

    Args:
        num_users (int): Number of unique users (train + test)
        num_items (int): Number of unique items (train + test)
        train_ratings (pd.DataFrame): Dataframe containing the movie ratings for training
        val_ratings (pd.DataFrame): Dataframe containing the movie ratings for validating
    """

    def __init__(self, num_users, num_items, train_ratings, val_ratings):
        super().__init__()
        self.dataset = MovieLensTrainDataset(train_ratings)
        self.validation_dataset = MovieLensTrainDataset(val_ratings)

        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=8)
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=8)

        input_size = 8 + 8  # user_embedded + item_embedded
        self.fc1 = nn.Linear(in_features=input_size, out_features=64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.bn2 = nn.BatchNorm1d(32)
        self.output = nn.Linear(in_features=32, out_features=11)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, user_input, item_input):
        # Pass through embedding layers
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)

        # Concat the two embedding layers
        vector = torch.cat([user_embedded, item_embedded], dim=-1)

        # Pass through dense layers
        # vector = nn.ReLU()(self.fc1(vector))
        # vector = nn.ReLU()(self.fc2(vector))
        vector = self.fc1(vector)
        vector = self.bn1(vector)  # Нормализация после первого слоя
        vector = nn.ReLU()(vector)
        vector = self.fc2(vector)
        vector = self.bn2(vector)  # Нормализация после второго слоя
        vector = nn.ReLU()(vector)
        pred = self.output(vector)

        return pred

    def training_step(self, batch, batch_idx):
        user_input, item_input, labels = batch
        labels = labels.long()  # Ensure that labels are in the correct type
        predicted_logits = self(user_input, item_input)

        loss = self.loss(predicted_logits, labels)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        user_input, item_input, labels = batch
        labels = labels.long()
        predicted_logits = self(user_input, item_input)

        loss = self.loss(predicted_logits, labels)

        self.log("val_loss", loss, prog_bar=True)
        return loss

    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
    #     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    #     return [optimizer], [scheduler]

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)

    def train_dataloader(self):
        return DataLoader(
            self.dataset, batch_size=512, num_workers=5, persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation_dataset,
            batch_size=512,
            num_workers=5,
            persistent_workers=True,
        )


if __name__ == "__main__":
    MAX_EPOCHS = 10

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.info("Считываем данные")
    ratings = pd.read_csv(constants.RATINGS_PATH, parse_dates=["timestamp"])

    # TAKE 30% HERE
    logging.info("Предобрабатываем данные")
    # rand_userIds = np.random.choice(ratings['userId'].unique(),
    #                             size=int(len(ratings['userId'].unique())*0.3),
    #                             replace=False)

    # ratings = ratings.loc[ratings['userId'].isin(rand_userIds)]

    train_ratings, test_ratings = train_test_split(ratings)

    # Init NCF model
    logging.info("Инициализируем модель")
    num_users = ratings["userId"].max() + 1
    num_items = ratings["movieId"].max() + 1

    model = NCF(num_users, num_items, train_ratings, test_ratings)

    checkpoint_callback = ModelCheckpoint(
        dirpath=r"src/weights/", filename="{epoch}-{val_loss:.2f}", monitor="val_loss"
    )

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        devices="auto",
        accelerator="auto",
        logger=False,
        callbacks=[checkpoint_callback],
        fast_dev_run=False,
    )

    logging.info("Запускаем обучение")
    trainer.fit(model)
    trainer.save_checkpoint(r"src/weights/NCF_FINAL_epoch.ckpt")
