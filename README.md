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

## API

It is possible to fetch our trained model deployed on Google Cloud Function by calling the following endpoint and specifying the image from the `test` set to predict.

```
curl "https://europe-west1-dtumlops-338418.cloudfunctions.net/als-inference?image=B_test.jpg"
```

## Development

Create a new conda enviroment by running: `conda env create --file=environment.yml`.

Therefore, enable it: `conda activate mlops`.

## Project Organization


    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── Dockerfile          <- Dockerfiles for building images
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │   └── conv_nn_copy.py        <- Architecture of our model for local deployement
    │   └── deploy_locally.py        <- Python file scripting the model
    │   └── deployable_model.pt        <- Scripted model
    │   └── instructions.txt       <- Torchserve lines to deploy the model locally
    │   └── model_store        <- Folder to store the deployable_model.mar file
    │    
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   ├── distributed_datas      <- reports from distributed data loading
    │   ├── quantization       <- Reports from compare_model.py : txt file with accuracy and computing time
    │   └── figures        <- Figures from Kornia tranform
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   ├── distributed_data_loading.py           <- Look at the different data loading
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── gcloud_functions       <- Scripts to deploy the model on gcloud
    │   │   ├── main.py
    │   │   └── requirements.txt.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── conv_nn.py       <- Main model
    │   │   ├── quantized_conv_nn.py       <- Quantized model
    │   │   ├── compare_model.py       <- Comparaisons of the model, quantized model and compiled model
    │   │   ├── kornia_transform.py       <- Kornia transformations
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   ├── profiling      <- Scripts to profile the cnn
    │   │   └── model_profiling.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    ├── tests         <- pytest running on the repository
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


## Authors

* Canevaro, Alessandro
* Hrafnkelsdóttir, Erla
* Kervellec, Loic Thibaut
* Rampazzo, Pietro
