import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid

from src.models.kornia_trans import transform

plt.rcParams["savefig.bbox"] = "tight"

warnings.filterwarnings("ignore", category=UserWarning)


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        to_np = np.asarray(img).transpose((1, 2, 0))
        axs[0, i].imshow(np.asarray(to_np))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()


if __name__ == "__main__":
    train = torch.load("data/processed/train.pt")
    torch.manual_seed(3)  # seed

    images = [train[i][0] for i in range(4)]
    trans = [transform(img).squeeze(0) for img in images]
    grid = make_grid(images + trans, nrow=4)
    show(grid)
