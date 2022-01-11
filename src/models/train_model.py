import torch
from conv_nn import ConvNet
from torch import nn, optim


def train(epochs = 20, lr = 0.01):
    print("Training")

    model = ConvNet()

    train = torch.load("data/processed/train.pt")
    train_set = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loss = []

    for e in range(epochs):
        running_loss = 0
        for images, labels in train_set:
            images = images.view(images.shape[0], -1)

            optimizer.zero_grad()
            outputs = model(images)

            labels = labels.long()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch: {e} - Training loss: {running_loss/len(train_set):5f}")
        train_loss.append(running_loss / len(train_set))


if __name__ == "__main__":
    train()