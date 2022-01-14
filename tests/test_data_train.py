import torch

data = torch.load("data/processed/train.pt") #datas

assert len(set([label for _, label in data])) == 3, "Not the good number of classes"

for i in range(0,len(data)): #good size
    assert data[i][0].shape == torch.Size([3, 224, 224]), "No the good image size"