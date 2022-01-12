import torch
from conv_nn import ConvNet
from torch import nn, optim


def train(epochs=20, lr=0.001):
    print("Training")

    model = ConvNet()

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

            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch: {e} - Training loss: {running_loss/len(train_set):5f}")
        train_loss.append(running_loss / len(train_set))

    torch.save(model.state_dict(), "models/trained_model.pt")


if __name__ == "__main__":
    train()
