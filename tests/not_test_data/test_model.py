import torch
from src.models.conv_nn import ConvNet


def test_model():
    data = torch.rand([64, 3, 224, 224]) # random datas
    model = ConvNet(1028,512)
    assert model(data).shape == torch.Size([64, 29]), "Not the good output"
