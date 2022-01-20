# from os import environ

import io

import torch
import torchvision.transforms.functional as TF
# import wandb
from conv_nn import ConvNet
# from omegaconf import OmegaConf
# Imports the Google Cloud client library
from google.cloud import storage
from PIL import Image


def get_data(bucket, path):
    # Instantiates a client
    client = storage.Client()

    # Retrieve an existing bucket
    bucket = client.get_bucket(bucket)

    # Then do other things...
    blob = bucket.get_blob(path)
    buffer = blob.download_as_string()

    # because model downloaded into string, need to convert it back
    return io.BytesIO(buffer)


def predict():
    model_buffer = get_data("dtumlopsdata", "models/trained_model.pt")
    data_buffer = get_data("dtumlopsdata", "data/raw/asl_alphabet_test/A_test.jpg")

    state_dict = torch.load(model_buffer)
    model = ConvNet(1024, 512)
    model.load_state_dict(state_dict)

    image = Image.open(data_buffer)
    x = TF.resize(image, 224)
    x = TF.to_tensor(x)
    x.unsqueeze_(0)
    # print(x.shape)

    print("evaluating")
    ps = torch.exp(model(x))
    top_p, top_class = ps.topk(1, dim=1)
    return top_p.item(), top_class.item()
