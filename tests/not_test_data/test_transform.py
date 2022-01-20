import torch

from src.models.kornia_trans import transform


def test_transform():
    data = torch.rand([64, 3, 224, 224])  # random datas
    assert transform(data).shape == data.shape, "Not the good shape"
    assert transform(data).values != data.values, "The same image"
