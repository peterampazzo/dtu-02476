import logging

import hydra
import torch
from conv_nn import ConvNet
from kornia_trans import transform
from omegaconf import OmegaConf
from torch import nn, optim

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="default.yaml")
def train(config):
    log.info("Training")
    log.info(f"configuration: \n {OmegaConf.to_yaml(config)}")
    hparams = config.experiment
    torch.manual_seed(hparams["seed"])
    lr = hparams["lr"]
    epochs = hparams["epochs"]
    out_features1 = hparams["out_features1"]
    out_features2 = hparams["out_features2"]

    model = ConvNet(out_features1, out_features2)

    train = torch.load("data/processed/train.pt")
    train_set = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
    model.train()

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loss = []

    for e in range(epochs):
        running_loss = 0
        for images, labels in train_set:

            optimizer.zero_grad()

            images = transform(images)  # kornia transformations

            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        log.info(f"Epoch: {e} - Training loss: {running_loss/len(train_set):5f}")
        train_loss.append(running_loss / len(train_set))
    torch.save(model.state_dict(), "models/trained_model.pt")


if __name__ == "__main__":
    train()
