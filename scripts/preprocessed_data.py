# Cached access to the four files that datasets_loading_preprocessing.py saves.
#
# Every experiment driver reads the same preprocessed datasets, so they are loaded
# once per process and handed out from here instead of being threaded through
# function parameters from main.py. Nothing in this module ever re-runs
# preprocessing: that downloads gigabytes and is a separate, deliberate step.


# General libraries
import json
import os
import sys
from functools import lru_cache

# General
import pandas as pd

# PyTorch:
import torch

# Project:
# DATASETS_PATH is imported rather than recomputed so that the module writing the
# files and the module reading them can never disagree about where they are.
from datasets_loading_preprocessing import DATASETS_PATH, HFMalNetDataset


# Global variables:
REQUIRED_DATASET_FILES = [
    "MalNet_datasets.pt",
    "Malnet_Original_splits.pt",
    "MalNet_datasets_df.csv",
    "datasets_statistics.json"
]


# MalNet_datasets.pt was written by running datasets_loading_preprocessing.py as a
# script, so its pickle records the class as "__main__.HFMalNetDataset" (verified by
# reading the GLOBAL opcodes out of the saved file). torch.load resolves that name
# against whatever module is __main__ at load time, which is why main.py imports the
# class at its very first line. Binding it here instead makes the load work from any
# entry point - and each experiment driver is now allowed to be the entry point.
setattr(sys.modules["__main__"], "HFMalNetDataset", HFMalNetDataset)


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


def strip_stored_num_nodes(datasets):
    """
    Removes the num_nodes attribute that only the Original dataset carries.

    Original is loaded by torch_geometric.datasets.MalNetTiny, which stores num_nodes
    on every graph. Common and Distinct come from the HuggingFace variant, which does
    not store it. PyG's collater takes the union of the keys in a batch and then
    demands every graph carry every one of them, so a batch that mixes Original with
    Distinct raises KeyError: 'num_nodes' - and a batch mixing exactly those two is
    what the k sweep trains on at every k above 0.

    Dropping it rather than adding it to the other two is lossless: Data.num_nodes
    falls back to x.size(0), and the stored value equals x.size(0) for all 5,000
    Original graphs. Doing it here, once, covers every consumer of the cached
    datasets, and it stays a no-op if preprocessing is ever changed to load all three
    datasets through the same loader.

    Input:
    --- datasets: a PyG dataset, or a dict nesting them to any depth.
    Output: None, the datasets are modified in place.
    """

    if isinstance(datasets, dict):

        for value in datasets.values():
            strip_stored_num_nodes(value)

        return None

    data = getattr(datasets, "_data", None)

    if data is not None and "num_nodes" in data:
        del data["num_nodes"]

    slices = getattr(datasets, "slices", None)

    if slices is not None:
        slices.pop("num_nodes", None)

    # The collated store also keeps a private per-graph list, and PyG's separate()
    # re-attaches num_nodes from it every time a graph is materialised. Removing the
    # public key alone therefore changes nothing.
    store = getattr(data, "_store", None)

    if store is not None and "_num_nodes" in store.__dict__:
        del store.__dict__["_num_nodes"]

    # The cached per-graph list, if one was built, still holds the old keys.
    datasets._data_list = None

    return None


@lru_cache(maxsize=None)
def get_preprocessed_datasets():
    """
    Validates and loads the four preprocessing outputs, once per process.

    The four objects together are about 1.6 GB, and a full experiment sweep asks for
    them hundreds of times, so the cache is what makes the accessors below free to
    call from anywhere.

    Input: None
    Output:
    --- the 4-tuple that load_preprocessed_datasets returns.
    """

    validate_preprocessed_datasets()

    preprocessed = load_preprocessed_datasets()

    # The three datasets have to be batchable together, and out of the box they are
    # not. See strip_stored_num_nodes.
    strip_stored_num_nodes(preprocessed[0])
    strip_stored_num_nodes(preprocessed[1])

    return preprocessed


# The four cached objects. Everything they return is shared by every caller in the
# process, so treat all of it as read-only: mutating a graph here changes it for every
# experiment that runs afterwards.

def get_datasets():
    """ {dataset_name: {malware_type: {split: PyG dataset_subset}}}. """
    return get_preprocessed_datasets()[0]

def get_original_splits():
    """ The MalNet-Tiny official splits, keyed by 'train', 'val' and 'test'. """
    return get_preprocessed_datasets()[1]

def get_dataframe():
    """ The graph-level pandas DataFrame the tabular models train on. """
    return get_preprocessed_datasets()[2]

def get_statistics():
    """ Per malware type statistics, as saved in datasets_statistics.json. """
    return get_preprocessed_datasets()[3]


# One dataset each, in the {malware_type: {split: PyG dataset_subset}} form that
# collect_graphs expects.

def get_original():
    """ MalNet-Tiny itself: 4 malware types plus benign. """
    return get_datasets()["Original"]

def get_common():
    """ The Common variant: the same 5 types as Original. """
    return get_datasets()["Common"]

def get_distinct():
    """ The Distinct variant: 5 malware types, no benign graphs at all. """
    return get_datasets()["Distinct"]
