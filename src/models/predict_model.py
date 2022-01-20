from os import environ
from string import ascii_uppercase

import click
import torch
import wandb
from conv_nn import ConvNet
from omegaconf import OmegaConf


@click.command()
@click.argument("profile", type=int, default=0)
def predict(profile: int):
    config = OmegaConf.load("config.yaml")
    out_features1 = config["profiles"][profile]["out_features1"]
    out_features2 = config["profiles"][profile]["out_features2"]

    model = ConvNet(out_features1, out_features2)
    model.load_state_dict(torch.load("models/trained_model.pt"))
    test_data = torch.load("data/processed/test.pt")
    test_set = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)

    environ["WANDB_API_KEY"] = config.wandb.api_key
    environ["WANDB_MODE"] = config.wandb.mode
    wandb.init(project=config.wandb.project, entity=config.wandb.entity)

    test_data_at = wandb.Artifact(
        "test_samples_" + str(wandb.run.name), type="predictions"
    )
    test_table = wandb.Table(columns=["image", "label", "class_prediction"])

    accuracies = []
    steps = 0
    label_map = {
        k: i for k, i in enumerate(list(ascii_uppercase) + ["del", "nothing", "space"])
    }
    print_every = 100

    with torch.no_grad():
        print("Evaluating")
        for images, labels in test_set:
            steps += 1
            if (steps % print_every == 0) or (steps == len(test_data)):
                print(f"Evaluating image {steps}/{len(test_data)}")

            output = model.forward(images)
            ps = torch.exp(output)
            equals = labels.data == ps.max(1)[1]

            batch_accuracy = torch.mean(equals.type(torch.FloatTensor))
            accuracies.append(batch_accuracy)

            test_table.add_data(
                wandb.Image(images),
                label_map[int(labels)],
                label_map[int(ps.max(1)[1])],
            )

    accuracy = sum(accuracies) / len(accuracies)

    wandb.log({"Test accuracy": accuracy.item()})
    print(f"Test accuracy: {accuracy*100}%")

    test_data_at.add(test_table, "predictions")
    wandb.run.log_artifact(test_data_at)


if __name__ == "__main__":
    predict()
