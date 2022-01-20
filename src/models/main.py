from os import environ

import torch
import wandb
from conv_nn import ConvNet
from omegaconf import OmegaConf
# Imports the Google Cloud client library
from google.cloud import storage
import io


def predict():
    print("Evaluating")

    model_buffer, data_buffer = get_data('dtumlopsdata', 'models/trained_model.pt', 'data/processed/test.pt')

    state_dict = torch.load(model_buffer)
    model = ConvNet(1024, 512)
    model.load_state_dict(state_dict)

    test_data = torch.load(data_buffer)
    test_set = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

    config = OmegaConf.load("config.yaml")
    #environ["WANDB_API_KEY"] = config.wandb.api_key
    #environ["WANDB_MODE"] = config.wandb.mode
    #wandb.init(project=config.wandb.project, entity=config.wandb.entity)

    accuracies = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(test_set):

            print(f"Evaluating Batch {i}/{len(test_data)//64}")

            ps = torch.exp(model(images))
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)

            batch_accuracy = torch.mean(equals.type(torch.FloatTensor))
            accuracies.append(batch_accuracy)

    accuracy = sum(accuracies) / len(accuracies)

    #wandb.log({"Test accuracy": accuracy.item()})
    print(f"Accuracy: {accuracy*100}%")


def get_data(bucket, model_path, data_path):
    # Instantiates a client
    client = storage.Client()

    # Retrieve an existing bucket
    bucket = client.get_bucket('dtumlopsdata')

    # Then do other things...
    model_blob = bucket.get_blob('models/trained_model.pt')
    model_buffer = model_blob.download_as_string()

    # because model downloaded into string, need to convert it back
    model_buffer = io.BytesIO(model_buffer)
    print("model loaded")

    # Then do other things...
    data_blob = bucket.get_blob('data/processed/test.pt')
    data_buffer = data_blob.download_as_string()

    # because model downloaded into string, need to convert it back
    data_buffer = io.BytesIO(data_buffer)
    print("data loaded")

    return model_buffer, data_buffer

if __name__ == "__main__":
    predict()