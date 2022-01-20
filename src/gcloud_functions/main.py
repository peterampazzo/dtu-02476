import io

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
# Imports the Google Cloud client library
from google.cloud import storage
from PIL import Image
from torch import nn


class ConvNet(nn.Module):
    def __init__(self, out_features1: int, out_features2: int):
        super(ConvNet, self).__init__()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.flatten = nn.Flatten()
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=36864, out_features=out_features1),
            nn.ReLU(),
            nn.Linear(in_features=out_features1, out_features=out_features2),
            nn.ReLU(),
            nn.Linear(in_features=out_features2, out_features=29),
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = self.flatten(x)
        x = self.linear_layers(x)
        x = F.log_softmax(x, dim=1)
        return x


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


def predict(request):
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
