# 02476 MLOps Project

### Overall goal of the project
The main goal of this project is to apply what we have learned in this course about MLOps to a simple machine/deep learning problem. We aim to make our whole pipeline as understandable and efficient as possible, using the tools we have been given - having good and clear structure and setup, adding comments to the code to make it easier to understand and so on. We will tackle an image classification problem and try to get as good results as possible, given our timeframe.

### What framework are you going to use
It was decided to use the [Kornia](https://github.com/kornia/kornia) framework as it was deemed the best suited for this project which involves image classification.

### How to you intend to include the framework into your project
The main purpose of the Kornia framework in this project is [data augmentation](https://kornia.readthedocs.io/en/latest/applications/image_augmentations.html). The goal is to obtain a larger dataset, adding transformed images to the dataset we have already. Specifically, the focus is on the augmentation, the color, and the enhance modules of Kornia as these will provide all the necessary functions to manipulate the images (i.e. rotating, scaling, normalizing, etc.).

### What data are you going to run on

> Kaggle link:

https://www.kaggle.com/grassknoted/asl-alphabet

> General description:

The data set is a collection of images of hands signing the American Sign Language alphabet, separated in 29 folders which represent the various classes.

> Size:

* Train data: 87,000 images, 200x200 pixels each.
* Test data: 29 images. 

> Classes:

* 26 for the letters A-Z.
* 3 classes for SPACE, DELETE and NOTHING.

### What deep learning models do you expect to use

We will use mostly CNN (convolutional neural networks). Our focus will be on the overall architecture and parameters of our network (number of layers, padding, strides, max-pooling etc.).

## Authors

* Canevaro, Alessandro
* Hrafnkelsdóttir, Erla
* Kervellec, Loic Thibaut
* Rampazzo, Pietro
