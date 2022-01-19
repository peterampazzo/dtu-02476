import logging
import sys
import warnings

import click
import torch
from conv_nn import ConvNet
from kornia_trans import transform
from omegaconf import OmegaConf
from torch import nn, optim

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


@click.command()
@click.argument(
    "input_filepath",
    default="data/processed",
    type=click.Path(exists=True),
)
@click.argument("output_filepath", default="models", type=click.Path())
@click.argument("profile", type=int, default=0)
def train(input_filepath: str, output_filepath: str, profile: int):
    config = OmegaConf.load("config.yaml")
    logger.info("Training")
    hparams = config["profiles"][profile]
    logger.info(f"configuration: \n {OmegaConf.to_yaml(hparams)}")
    torch.manual_seed(hparams["seed"])
    lr = hparams["lr"]
    epochs = hparams["epochs"]
    out_features1 = hparams["out_features1"]
    out_features2 = hparams["out_features2"]

    model = ConvNet(out_features1, out_features2)

    train_data = torch.load(f"{input_filepath}/train.pt")
    train_set = torch.utils.data.DataLoader(
        train_data,
        batch_size=hparams["batch_size"],
        shuffle=True,
        num_workers=hparams["num_workers"],
    )
    model.train()

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loss = []

    for e in range(epochs):
        logger.info(f"Starting epoch: {e+1}/{epochs}")
        running_loss = 0
        for i, (images, labels) in enumerate(train_set):

            logger.info(f"    Batch {i}/{len(train_data)//hparams['batch_size']}")

            optimizer.zero_grad()

            images = transform(images)  # kornia transformations

            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        logger.info(
            f"Finished epoch: {e+1} - Training loss: {running_loss/len(train_set):5f}"
        )
        train_loss.append(running_loss / len(train_set))

    torch.save(model.state_dict(), f"{output_filepath}/trained_model.pt")
    logger.info("Model saved")


if __name__ == "__main__":
    train()
    logger.info("Completed")
