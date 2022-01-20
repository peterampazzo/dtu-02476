import torch
import torch.nn.functional as F
from torch import nn


class ConvNet_quantized(nn.Module):
    def __init__(self, out_features1: int, out_features2: int):
        super(ConvNet_quantized, self).__init__()

        self.quant = torch.quantization.QuantStub()

        self.quant = torch.quantization.QuantStub()
        self.conv1 = torch.nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=8, stride=4
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=4, stride=2
        )
        self.conv3 = torch.nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1
        )
        self.relu = torch.nn.ReLU()
        self.dequant = torch.quantization.DeQuantStub()

        self.dequant = torch.quantization.DeQuantStub()

        self.flatten = nn.Flatten()
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=36864, out_features=out_features1),
            nn.ReLU(),
            nn.Linear(in_features=out_features1, out_features=out_features2),
            nn.ReLU(),
            nn.Linear(in_features=out_features2, out_features=29),
        )

    # Defining the forward pass
    def forward(self, x):

        x = self.quant(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)

        x = self.flatten(x)
        x = self.linear_layers(x)
        x = self.dequant(x)
        x = F.log_softmax(x, dim=1)

        return x
