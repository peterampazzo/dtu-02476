import torch
from src.models.conv_nn import ConvNet


def test_model_output():
    data = torch.rand([64, 3, 224, 224]) # random datas
    model = ConvNet(1028,512)
    assert model(data).shape == torch.Size([64, 29]), "Not the good output"

def test_model_layers():
    model = ConvNet(1028,512)
    all_layer=[]
    for i in model.modules():
        all_layer+=[i]
    nb_layer=len(all_layer)-2
    assert nb_layer==13, "Not the good number of layer"

