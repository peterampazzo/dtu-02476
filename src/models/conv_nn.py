import torch
from torch import nn

class ConvNet(nn.Module):   
    def __init__(self):
        super(ConvNet, self).__init__()

        self.cnn_layers = None

        self.linear_layers = None

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

if __name__ == '__main__':
    cnn = ConvNet()
    print('All done!')