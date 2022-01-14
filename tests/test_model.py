import torch
from src.models.conv_nn import ConvNet
import random
import os
import pytest

@pytest.mark.skipif(not os.path.exists('data/processed/test.pt'), reason="No Datas")

def test_model():
    data = torch.load("data/processed/train.pt") 
    data = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)#datas
    model = ConvNet(1028,512)
    assert model(iter(data).next()[0]).shape == torch.Size([64, 29]), "Not the good output"