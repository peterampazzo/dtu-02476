import os
import random

import pytest
import torch


@pytest.mark.skipif(not os.path.exists("data/processed/test.pt"), reason="No Datas")
def test_length():
    data = torch.load("data/processed/test.pt")  # datas
    assert (
        len(set([label for _, label in data])) == 29
    ), "Not the good number of classes"


def test_shape():
    data = torch.load("data/processed/test.pt")  # datas
    randomlist = random.sample(range(0, len(data) - 1), 200)
    for i in randomlist:  # good size
        assert data[i][0].shape == torch.Size([3, 224, 224]), "No the good image size"
