import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F

class ConvNet(nn.Module):   
    def __init__(self):
        super(ConvNet, self).__init__()

        self.cnn_layers = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4), nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2), nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1), nn.ReLU())

        self.flatten = nn.Flatten()
        self.linear_layers = nn.Sequential(nn.Linear(in_features=36864, out_features=1024),nn.ReLU(),
        nn.Linear(in_features=1024, out_features=512),nn.ReLU(),nn.Linear(in_features=512, out_features=3))

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = self.flatten(x)
        x = self.linear_layers(x)
        x = F.log_softmax(x, dim=1)
        return x

if __name__ == '__main__':
    cnn = ConvNet()
    print('All done!')