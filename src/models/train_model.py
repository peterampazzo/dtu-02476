import logging
import warnings

import torch
from conv_nn import ConvNet
from kornia_trans import transform
from omegaconf import OmegaConf
from torch import nn, optim

warnings.filterwarnings("ignore", category=UserWarning)

log = logging.getLogger(__name__)


def train():
    config = OmegaConf.load("config.yaml")
    print("Training")
    print(f"configuration: \n {OmegaConf.to_yaml(config)}")
    hparams = config["hyperparameters"]
    torch.manual_seed(hparams["seed"])
    lr = hparams["lr"]
    epochs = hparams["epochs"]
    out_features1 = hparams["out_features1"]
    out_features2 = hparams["out_features2"]

    model = ConvNet(out_features1, out_features2)

    train_data = torch.load("data/processed/train.pt")
    train_set = torch.utils.data.DataLoader(
        train_data, batch_size=64, shuffle=True, num_workers=0
    )
    model.train()

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loss = []

    for e in range(epochs):
        print(f"Starting epoch: {e+1}/{epochs}")
        running_loss = 0
        for i, (images, labels) in enumerate(train_set):

            print(f"    Batch {i}/{len(train_data)//64}")

            optimizer.zero_grad()

            images = transform(images)  # kornia transformations

            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(
            f"Finished epoch: {e+1} - Training loss: {running_loss/len(train_set):5f}"
        )
        train_loss.append(running_loss / len(train_set))

    torch.save(model.state_dict(), "models/trained_model.pt")
    print("Model saved")


if __name__ == "__main__":
    train()
    print("All done")
