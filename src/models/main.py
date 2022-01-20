from os import environ

import torch
import wandb
from conv_nn import ConvNet
from omegaconf import OmegaConf



def predict():
    print("Evaluating")

    model = ConvNet(1024, 512)
    model.load_state_dict(torch.load("models/trained_model.pt"))
    test_data = torch.load("data/processed/test.pt")
    test_set = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

    config = OmegaConf.load("config.yaml")
    environ["WANDB_API_KEY"] = config.wandb.api_key
    environ["WANDB_MODE"] = config.wandb.mode
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


if __name__ == "__main__":
    # Imports the Google Cloud client library
    from google.cloud import storage
    import io

    # Instantiates a client
    client = storage.Client()

    # Retrieve an existing bucket
    bucket = client.get_bucket('dtumlopsdata')

    # Then do other things...
    blob = bucket.get_blob('models/trained_model.pt')
    buffer = blob.download_as_string()

    # because model downloaded into string, need to convert it back
    buffer = io.BytesIO(buffer)
    state_dict = torch.load(buffer)
    model = ConvNet(1024, 512)
    model.load_state_dict(state_dict)

    print("model loaded")

    # Then do other things...
    blob = bucket.get_blob('data/processed/test.pt')
    buffer = blob.download_as_string()

    # because model downloaded into string, need to convert it back
    buffer = io.BytesIO(buffer)
    test_data = torch.load(buffer)
    test_set = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)
    print("data loaded")