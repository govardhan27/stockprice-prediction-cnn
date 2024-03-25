from functools import partial

import pandas as pd
import torch
from torch.utils.data import DataLoader

from model.dataset import CNNStocksDataset
from model.model import CNNStocksModule


LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-6


def train(x_df: pd.DataFrame, y_series: pd.Series, epochs: int = 100):
    """
    Trains the CNNStocksModule model
    :param x_df: Inputs consisting of sequences of stock price returns
    :param y_series: Targets consisting of returns some days in advance of the reference dates
    :param epochs: Number of complete iterations to go through the data in order to train
    :return: The trained CNNStocksModule model
    """
    # Put data into GPU if possible
    dataloader_kwargs = {}
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')  # Store all training data in the GPU
        dataloader_kwargs['generator'] = torch.Generator(device='cuda')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    # Turn pandas objects into Pytorch tensor objects
    x_tensor, y_tensor = torch.tensor(x_df.values).float(), torch.tensor(y_series.values).float()

    # Set up the dataloader
    train_dataset = CNNStocksDataset(x_tensor, y_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, **dataloader_kwargs)

    loss_func = partial(torch.nn.functional.huber_loss, delta=0.02)

    model = CNNStocksModule(x_df.shape[1]).train()
    if torch.cuda.is_available():  # Train on GPU if possible
        model = model.cuda()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # Conduct training which consists of homing the model in on the best parameters that minimize the loss
    for i in range(epochs):
        total_loss = 0.
        for x, y in train_dataloader:
            optimizer.zero_grad()
            out = model(x)
            loss = loss_func(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.cpu().detach().numpy()

    return model


def predict(trained_model, x_df: pd.DataFrame) -> pd.Series:
    """
    Generates predictions using a trained model
    :param trained_model: Trained Pytorch model
    :param x_df: Inputs to generate predictions for
    :return: Series containing predictions, with reference dates as indices
    """
    trained_model.eval()

    x_tensor = torch.tensor(x_df.values).float()
    prediction = trained_model(x_tensor)

    return pd.Series(prediction.cpu().detach().numpy(), index=x_df.index)
