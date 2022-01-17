import logging
from pathlib import Path

import click
import torch
from dotenv import find_dotenv, load_dotenv
from torchvision import datasets, transforms


def load_data(root_dir: str, output_filepath: str) -> None:
    """
    Generate and save train and test dataloader.

            Parameters:
                    root_dir (str): Location raw data
                    output_filepath (str): Location output files

            Returns:
                    None
    """
    test_size = 0.2
    data_transforms = transforms.Compose(
        [transforms.Resize(224), transforms.ToTensor()]
    )

    dataset = datasets.ImageFolder(root_dir, transform=data_transforms)

    torch.manual_seed(1)
    dataset_size = len(dataset)
    test_size = int(test_size * dataset_size)
    train_size = dataset_size - test_size

    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    logging.info(f"Training size: {len(train_dataset)}")
    logging.info(f"Test size: {len(test_dataset)}")

    torch.save(train_dataset, f"{output_filepath}/train.pt")
    torch.save(test_dataset, f"{output_filepath}/test.pt")


@click.command()
@click.argument(
    "input_filepath",
    default="data/raw/asl_alphabet_train",
    type=click.Path(exists=True),
)
@click.argument("output_filepath", default="data/processed/", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    load_data(input_filepath, output_filepath)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
