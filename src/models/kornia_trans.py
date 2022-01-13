import kornia
from torch import nn

transform = nn.Sequential(
    kornia.augmentation.ColorJitter(brightness=0.5,
                                    contrast=0.5, 
                                    saturation=0.5, 
                                    hue=1.0, 
                                    p=0.5),
    kornia.augmentation.RandomAffine((-30, 30),
                                     p=0.5),
    kornia.augmentation.RandomElasticTransform(kernel_size=(63, 63), 
                                               sigma=(32.0, 32.0), 
                                               alpha=(1.0, 1.0),
                                               p=1),
    kornia.augmentation.RandomGaussianNoise(mean=0.0, 
                                            std=0.2, 
                                            p=0.1),
    kornia.augmentation.RandomThinPlateSpline(scale=0.2, 
                                              p=0.1)
)