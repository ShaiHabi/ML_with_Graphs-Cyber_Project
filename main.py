# Required for loading the saved MalNet dataset objects:
from scripts.datasets_loading_preprocessing import HFMalNetDataset


import os
import random
import sys
import numpy as np
import torch

# Global variables:
RANDOM_STATE = 42
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
DATASETS_PATH = os.path.join(PROJECT_PATH, "datasets")
REQUIRED_DATASET_FILES = [
    "MalNet_datasets.pt",
    "Malnet_Original_splits.pt",
    "MalNet_datasets_df.csv",
    "datasets_statistics.json"
]


def set_random_seed(seed):
    """
    Sets the random seed for reproducible experiments.

    Input:
    --- seed: int
    Output: None
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Makes CUDA operations more reproducible:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return None


def validate_preprocessed_datasets():
    """
    Checks that the datasets directory and all four preprocessing output files exist.
    Raises FileNotFoundError if the directory or one of the files is missing.

    Input: None
    Output: None
    """

    print("Validating preprocessed datasets ...", end="")
    if not os.path.isdir(DATASETS_PATH):
        raise FileNotFoundError(
            "\nThe datasets directory does not exist.\n"
            "Run the following command first:\n"
            "python scripts/datasets_loading_preprocessing.py"
        )

    missing_files = []

    for file_name in REQUIRED_DATASET_FILES:
        file_path = os.path.join(DATASETS_PATH, file_name)

        if not os.path.isfile(file_path):
            missing_files.append(file_name)

    if missing_files:
        missing_files_text = "\n".join(
            f"--- {file_name}" for file_name in missing_files
        )

        raise FileNotFoundError(
            "\nThe datasets directory is missing the following files:\n"
            f"{missing_files_text}\n\n"
            "Run the following command first:\n"
            "python scripts/datasets_loading_preprocessing.py"
        )

    print("Done!")
    return None


def load_preprocessed_datasets():
    """
    Loads the four preprocessing output files.

    Input: None
    Outputs:
    --- MalNet_datasets
    --- Malnet_Original_splits
    --- MalNet_datasets_df
    --- datasets_statistics
    """

    import json
    import pandas as pd

    malnet_datasets_path = os.path.join(DATASETS_PATH, "MalNet_datasets.pt")
    malnet_original_splits_path = os.path.join(DATASETS_PATH, "Malnet_Original_splits.pt")
    malnet_datasets_df_path = os.path.join(DATASETS_PATH, "MalNet_datasets_df.csv")
    statistics_path = os.path.join(DATASETS_PATH, "datasets_statistics.json")

    print("Loading preprocessed datasets ...", end="")
    MalNet_datasets = torch.load(malnet_datasets_path, weights_only=False)
    Malnet_Original_splits = torch.load(malnet_original_splits_path, weights_only=False)
    MalNet_datasets_df = pd.read_csv(malnet_datasets_df_path)

    with open(statistics_path, "r", encoding="utf-8") as file:
        datasets_statistics = json.load(file)

    print("Done!")

    return (
        MalNet_datasets,
        Malnet_Original_splits,
        MalNet_datasets_df,
        datasets_statistics
    )


def main():
    """ Runs the project's experiments using the preprocessed datasets. """

    # Sets the random seed:
    set_random_seed(RANDOM_STATE)

    # Selects the available device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Random seed:", RANDOM_STATE)
    print("Device:", device)

    try:
        validate_preprocessed_datasets()
    except FileNotFoundError as error:
        print(error)
        sys.exit(1) # Error

    (MalNet_datasets,
     Malnet_Original_splits,
     MalNet_datasets_df,
     datasets_statistics) = load_preprocessed_datasets()

    # The GNN experiments will be added here.



    return 0


if __name__ == "__main__":
    main()