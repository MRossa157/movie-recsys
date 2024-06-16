import logging

import constants
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from utils import train_test_split


class MovieLensTrainDataset(Dataset):
    """MovieLens PyTorch Dataset for Training

    Args:
        ratings (pd.DataFrame): Dataframe containing the movie ratings
        all_movieIds (list): List containing all movieIds
        is_training (bool): Default is True. Indicate for progress bar

    """

    def __init__(self, ratings, all_movieIds, is_training: bool = True):
        self.is_training = is_training
        self.users, self.items, self.labels = self.get_dataset(ratings, all_movieIds)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]

    def get_dataset(self, ratings, all_movieIds):
        users, items, labels = [], [], []
        user_item_set = set(zip(ratings["userId"], ratings["movieId"]))

        num_negatives = 4
        for u, i in tqdm(
            user_item_set,
            desc=f"Generating negative sample for {'training' if self.is_training else 'validating'}",
        ):
            users.append(u)
            items.append(i)
            labels.append(1)
            for _ in range(num_negatives):
                negative_item = np.random.choice(all_movieIds)
                while (u, negative_item) in user_item_set:
                    negative_item = np.random.choice(all_movieIds)
                users.append(u)
                items.append(negative_item)
                labels.append(0)

        return torch.tensor(users), torch.tensor(items), torch.tensor(labels)


class NCF(pl.LightningModule):
    """Neural Collaborative Filtering (NCF)

    Args:
        num_users (int): Number of unique users
        num_items (int): Number of unique items
        ratings (pd.DataFrame): Dataframe containing the movie ratings for training
        all_movieIds (list): List containing all movieIds (train + test)
    """

    def __init__(self, num_users, num_items, ratings, all_movieIds):
        super().__init__()

        self.ratings = ratings
        self.all_movieIds = all_movieIds

        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=8)
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=8)
        self.fc1 = nn.Linear(in_features=16, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.output = nn.Linear(in_features=32, out_features=1)

    def forward(self, user_input, item_input):
        # Pass through embedding layers
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)

        # Concat the two embedding layers
        vector = torch.cat([user_embedded, item_embedded], dim=-1)

        # Pass through dense layer
        vector = nn.ReLU()(self.fc1(vector))
        vector = nn.ReLU()(self.fc2(vector))

        # Output layer
        pred = nn.Sigmoid()(self.output(vector))

        return pred

    def training_step(self, batch, batch_idx):
        user_input, item_input, labels = batch
        predicted_labels = self(user_input, item_input)
        loss = nn.BCELoss()(predicted_labels, labels.view(-1, 1).float())
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def train_dataloader(self):
        return DataLoader(
            MovieLensTrainDataset(self.ratings, self.all_movieIds, is_training=True),
            batch_size=constants.NCF_BATCH_SIZE,
            num_workers=5,
            persistent_workers=True,
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.info("Считываем данные")
    ratings = pd.read_csv(constants.RATINGS_PATH, parse_dates=["timestamp"])

    # TAKE 30% HERE
    # logging.info("Предобрабатываем данные")
    # rand_userIds = np.random.choice(
    #     ratings["userId"].unique(),
    #     size=int(len(ratings["userId"].unique()) * 0.3),
    #     replace=False,
    # )

    # ratings = ratings.loc[ratings["userId"].isin(rand_userIds)]

    train_ratings, _ = train_test_split(ratings)

    # Explicit to implicit convert
    train_ratings.loc[:, "rating"] = 1

    # Init NCF model
    logging.info("Инициализируем модель")
    num_users = ratings["userId"].max() + 1
    num_items = ratings["movieId"].max() + 1
    all_movieIds = ratings["movieId"].unique()

    model = NCF(num_users, num_items, train_ratings, all_movieIds)

    checkpoint_callback = ModelCheckpoint(
        dirpath=constants.WEIGHTS_PATH,
        filename="{epoch}-{train_loss:.2f}",
        monitor="train_loss",
    )

    trainer = pl.Trainer(
        # fast_dev_run=True,
        max_epochs=constants.NCF_MAX_EPOCHS,
        reload_dataloaders_every_n_epochs=1,
        devices="auto",
        accelerator="auto",
        logger=False,
        callbacks=[checkpoint_callback],
    )

    logging.info("Запускаем обучение")
    trainer.fit(model)

    logging.info(
        f"Сохраняем веса модели (последний чекпоинт) в папку {constants.WEIGHTS_PATH}"
    )
    trainer.save_checkpoint(
        f"{constants.WEIGHTS_PATH}/NCF_result_epochs={constants.NCF_MAX_EPOCHS}.ckpt"
    )
