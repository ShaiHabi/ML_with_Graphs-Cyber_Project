# By: Shai Habi, Tomer Gal Netser, 26-27/08/2026
# This code contains the implementation of the data processing: Data Loading and Data preprocessing.
# The results are automatically saved in "datasets" directory.


# Setup
## General libraries
import pandas as pd
import numpy as np
import json
from pathlib import Path

## PyTorch:
import torch

## Datasets - for malnet-tiny:
from torch_geometric.datasets import MalNetTiny
import os.path as osp
from collections import defaultdict

## HuggingFace
# safetensors is required by mntf.py, the loading script shipped with the dataset
from huggingface_hub import hf_hub_download
import os
import sys
import importlib
from torch_geometric.utils import degree
from torch_geometric.data import InMemoryDataset

## Predefined global variables
RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)

# Project paths
SCRIPT_PATH = Path(__file__).resolve().parent
PROJECT_PATH = SCRIPT_PATH.parent

DATASETS_PATH = PROJECT_PATH / "datasets"
RAW_DATASETS_PATH = DATASETS_PATH / "raw_src"

DATASETS_PATH.mkdir(parents=True, exist_ok=True)
RAW_DATASETS_PATH.mkdir(parents=True, exist_ok=True)


### Subsection 1: Data loading and Feature Engineering ###
# The following functions uploads the datasets and add two features for each node.

def load_MalNetTinyOriginal():
    """
    Loads the MalNet-Tiny datasets.
    As the source code does not split by malware_type, we do it ourselves.

    URL:
    https://pytorch-geometric.readthedocs.io/en/2.5.3/generated/torch_geometric.datasets.MalNetTiny.html

    Input: None
    Output:
    --- malnet_tiny: nested dict in the form
        {malware_type: {split: PyG dataset_subset}}.
    --- malnet_tiny_by_split: dict where the keys are ['train', 'val', 'test'].
    """

    # Loads all 5,000 processed graphs:
    root = os.path.join(RAW_DATASETS_PATH, "MalNetTiny_Original")
    full_malnet_tiny = MalNetTiny(root)


    # Adds log-transformed in-degree and out-degree features:
    all_features = []
    node_slices = [0]

    for index in range(len(full_malnet_tiny)):
          graph = full_malnet_tiny.get(index)

          source, target = graph.edge_index
          in_degree = degree(target, num_nodes=graph.num_nodes, dtype=torch.float)
          out_degree = degree(source, num_nodes=graph.num_nodes, dtype=torch.float)

          # Data Transformation by using log1p
          node_features = torch.stack([in_degree, out_degree], dim=1)
          node_features = torch.log1p(node_features)
          all_features.append(node_features)

          node_slices.append(node_slices[-1] + graph.num_nodes)


    # Stores the features inside the full dataset:
    full_malnet_tiny._data.x = torch.cat(all_features, dim=0)
    full_malnet_tiny.slices["x"] = torch.tensor(node_slices, dtype=torch.long)


    # Splits to malware types and original train/val/test split:
    split_path = osp.join(root, "raw", "split_info_tiny", "type")

    filenames = []
    split_names = []

    for split in ["train", "val", "test"]:
        with open(osp.join(split_path, f"{split}.txt")) as f:
          split_filenames = f.read().split("\n")[:-1] # As in the source code

        filenames += split_filenames
        split_names += [split] * len(split_filenames)

    types = [filename.split("/")[0] for filename in filenames]
    assert len(types) == len(full_malnet_tiny)
    assert len(split_names) == len(full_malnet_tiny)

    # For later creating the dataframes, we save the splits
    split_to_id = {"train": 0, "val": 1, "test": 2}
    official_split = torch.tensor(
          [split_to_id[split] for split in split_names],
          dtype=torch.long
    )

    # Creating the dictionaries
    indices_by_type_and_split = defaultdict(lambda: defaultdict(list))
    indices_by_split = defaultdict(list)

    binary_labels = torch.empty(len(full_malnet_tiny), dtype=torch.long)

    for index, malware_type in enumerate(types):
          # Preserve the original type name
          indices_by_type_and_split[malware_type][split_names[index]].append(index)

          # Preserve the original train/validation/test split
          indices_by_split[split_names[index]].append(index)

          # Assign the binary classification label
          binary_labels[index] = (0 if malware_type == "benign" else 1)


    # Change only the labels used by the model
    full_malnet_tiny._data.y = binary_labels

    # Preserve original train/val/test split as graph-level metadata
    # It's for the future dataframe
    full_malnet_tiny._data.official_split = official_split
    full_malnet_tiny.slices["official_split"] = torch.arange(
        len(full_malnet_tiny) + 1,
        dtype=torch.long
    )

    full_malnet_tiny._data_list = None

    # Preserve separate subsets by original malware type and official split
    malnet_tiny = {
          malware_type: {
              split: full_malnet_tiny[indices]
              for split, indices in split_indices.items()
          }
          for malware_type, split_indices
          in indices_by_type_and_split.items()
    }

    # Preserve separate subsets by original train/val/test split
    malnet_tiny_by_split = {
          split: full_malnet_tiny[indices]
          for split, indices
          in indices_by_split.items()
    }

    return malnet_tiny, malnet_tiny_by_split


class HFMalNetDataset(InMemoryDataset):
    """
    Holds an already collated (data, slices) pair.
    We keep this class instead of returning the upstream MNTF object, so that
    the datasets we torch.save stay loadable without mntf.py being importable.
    """
    def __init__(self, data, slices):
        super().__init__(root=None)
        self._data = data
        self.slices = slices


# The dataset was reworked upstream, so the old repository "nntvu/MalNet-Tiny-Features"
# (a data.pt + hash_list.pt pair) no longer applies. Each variant now stores its graphs
# and their malware type label in <variant>/base.safetensors, its official split in
# <variant>/splits.json, and the repository ships its own loading script, mntf.py.
MNTF_REPO_ID = "ngoctnq/malnet-features"
MNTF_ROOT = os.path.join(DATASETS_PATH, "raw_malNetTiny_features")

# The upstream variant name of each of our dataset names.
MNTF_VARIANTS = {"Common": "common", "Distinct": "distinct", "Original": "tiny"}

# The malware type behind every label id. y is the index of the type folder the graph
# came from, so it already encodes the malware type and the hash -> type mapping we
# used to download from GitHub is no longer needed. Copied from `raw_file_names` in
# github.com/ngoctnq/malnet-features/blob/master/training/graphgps/loader/dataset/malnet_tiny_features.py
#
# Note that "Distinct" is built from five malware types and holds no benign graph at
# all, so every one of its graphs gets the binary label 1 below.
MNTF_TYPE_NAMES = {
    "tiny":     ["addisplay", "adware", "benign", "downloader", "trojan"],
    "common":   ["addisplay", "adware", "benign", "downloader", "trojan"],
    "distinct": ["clicker++trojan", "malware", "riskware", "spr", "spyware"],
}


def get_MNTF_class():
    """
    Downloads the loading script shipped with the dataset and returns its MNTF class.

    URL:
    https://huggingface.co/datasets/ngoctnq/malnet-features/blob/main/mntf.py

    Input: None
    Output:
    --- MNTF : the upstream InMemoryDataset subclass.
    """

    script_path = hf_hub_download(
          repo_id=MNTF_REPO_ID,
          repo_type="dataset",
          filename="mntf.py",
          local_dir=MNTF_ROOT,
    )

    # Import it by adding its folder to the path
    script_dir = os.path.dirname(script_path)
    if script_dir not in sys.path:
          sys.path.insert(0, script_dir)

    return importlib.import_module("mntf").MNTF


def add_log_degree_features(dataset):
    """
    Replaces the node features of a collated dataset with
    [log1p(in_degree), log1p(out_degree)]. The dataset is modified in place.

    Input:
    --- dataset : PyG InMemoryDataset
    Output: None
    """

    data, slices = dataset._data, dataset.slices

    # edge_index is stored per graph, so its node ids are local to each graph.
    # Lifting them to global ids lets us count the degrees of all 5,000 graphs with
    # a single bincount: no edge ever crosses a graph boundary, so the global counts
    # are exactly the per-graph ones.
    edge_counts = slices["edge_index"][1:] - slices["edge_index"][:-1]
    per_edge_offset = torch.repeat_interleave(slices["x"][:-1], edge_counts)
    global_edge_index = data.edge_index + per_edge_offset.unsqueeze(0)

    num_nodes = int(slices["x"][-1])
    in_degree = torch.bincount(global_edge_index[1], minlength=num_nodes)
    out_degree = torch.bincount(global_edge_index[0], minlength=num_nodes)

    # Data Transformation by using log1p
    node_features = torch.stack([in_degree, out_degree], dim=1).float()
    data.x = torch.log1p(node_features)

    dataset._data_list = None
    return None # Changes are in-place


def load_HF_dataset(dataset_name, remove_isolated=True):
    """
    Loads MalNet-Tiny Common or Distinct, adds degree features once,
    and returns a dict of PyG dataset subsets by malware type.

    URLs:
    https://huggingface.co/datasets/ngoctnq/malnet-features
    https://github.com/ngoctnq/malnet-features

    Input:
    --- dataset_name : string, one of "Common", "Distinct" or "Original".
    --- remove_isolated : bool. Drops every node whose only edge is a self loop,
        which is what the dataset we used before the rework had already done.
        Pass False to keep all the nodes, like load_MalNetTinyOriginal does.
    Output:
    --- dataset : nested dict in the form
        {malware_type: {split: PyG dataset_subset}}.
    """

    assert dataset_name in MNTF_VARIANTS, f"Invalid dataset name: {dataset_name}"
    variant = MNTF_VARIANTS[dataset_name]

    MNTF = get_MNTF_class()

    # collator="none" asks for base.safetensors and splits.json only, which carry the
    # graph structure, the malware type labels and the official split. The semantic
    # and LLM feature files are 6-40GB each and we do not need them, since our node
    # features are the degrees computed below.
    mntf_dataset = MNTF(
        collator="none",
        ablation="base",
        variant=variant,
        llm_name=None,
        root=MNTF_ROOT,
        remove_isolated=remove_isolated,
    )

    # Hand the collated tensors over to our own class
    full_dataset = HFMalNetDataset(mntf_dataset._data, mntf_dataset.slices)
    splits = mntf_dataset.splits

    # Adds log-transformed in-degree and out-degree features:
    add_log_degree_features(full_dataset)

    # For later creating the dataframes, we save the splits.
    # The script names the validation split "valid", we name it "val".
    split_to_id = {"train": 0, "valid": 1, "test": 2}
    official_split = torch.empty(len(full_dataset), dtype=torch.long)

    for split_name, split_indices in splits.items():
        official_split[torch.tensor(split_indices, dtype=torch.long)] = split_to_id[split_name]

    # y holds the malware type id, so keep it before overwriting the labels
    type_names = MNTF_TYPE_NAMES[variant]
    type_ids = full_dataset._data.y.clone()

    # Preserve graph indices by their original malware type,
    # while creating binary labels for classification.
    indices_by_type_and_split = defaultdict(lambda: defaultdict(list))

    split_id_to_name = {
        0: "train",
        1: "val",
        2: "test"
    }

    for index, type_id in enumerate(type_ids.tolist()):
        # Preserve the original malware type name.
        malware_type = type_names[type_id]
        split_id = int(official_split[index].item())
        split_name = split_id_to_name[split_id]
        indices_by_type_and_split[malware_type][split_name].append(index)

    # benign = 0, malware = 1.
    # "Distinct" holds no benign type, so all of its graphs are labelled 1.
    benign_id = type_names.index("benign") if "benign" in type_names else -1
    binary_labels = (type_ids != benign_id).long()

    # Change only the labels used by the model.
    full_dataset._data.y = binary_labels

    # Preserve original train/val/test split as graph-level metadata
    # It's for the future dataframe
    full_dataset._data.official_split = official_split
    full_dataset.slices["official_split"] = torch.arange(
      len(full_dataset) + 1,
      dtype=torch.long
    )

    full_dataset._data_list = None

    # Preserve separate subsets by original malware type and official split.
    dataset = {
        malware_type: {
            split: full_dataset[indices]
            for split, indices in split_indices.items()
        }
        for malware_type, split_indices
        in indices_by_type_and_split.items()
    }

    return dataset


def load_MalNet_datasets():
    """
    Loads the MalNet-Tiny datasets.
    As the source code does not split by malware_type, we do it ourselves.
    Each dataset contains log1p transformation on in-degree and out-degree features.

    Input: None
    Output:
    --- MalNet_datasets : nested dict in the form
        {dataset_name: {malware_type: {split: PyG dataset_subset}}}.
    --- MalnetTiny_original_splits: dict where the keys are ['train', 'val', 'test'].
    """

    MalNet_datasets = {}

    for dataset_name in ["Original", "Common", "Distinct"]:
        print(f"Loading {dataset_name} ... ", end="")
        if dataset_name == "Original":
          malnet_tiny_by_type, MalnetTiny_original_splits = load_MalNetTinyOriginal()
          MalNet_datasets["Original"] = malnet_tiny_by_type
        else:
          MalNet_datasets[dataset_name] = load_HF_dataset(dataset_name)
        print("Done!")

    return MalNet_datasets, MalnetTiny_original_splits


def validate_MalNet_datasets_structure(MalNet_datasets):
    """
    Validates that every malware type in every dataset contains the
    official train/validation/test split with 700/100/200 graphs.

    Input:
    --- MalNet_datasets : nested dict in the form
        {dataset_name: {malware_type: {split: PyG dataset_subset}}}.
    Output: None
    """

    expected_split_sizes = {
        "train": 700,
        "val": 100,
        "test": 200
    }

    for dataset_name, malware_types in MalNet_datasets.items():
        for malware_type, splits in malware_types.items():

            actual_split_sizes = {
                split: len(graphs)
                for split, graphs in splits.items()
            }

            if actual_split_sizes != expected_split_sizes:
                raise ValueError(
                    f"Unexpected split sizes for "
                    f"{dataset_name}/{malware_type}: "
                    f"{actual_split_sizes}. Expected: "
                    f"{expected_split_sizes}."
                )

    return None




### Subsection 2: Data Statistics ###
# The following  code calculates statistics about the graphs in each malware type.

def malware_graph_statistics(dataset):
    """
    Computes graph-level statistics for a PyG InMemoryDataset.

    Input:
    --- dataset : dict of PyG subsets, where the keys are
        ['train', 'val', 'test'].
    Output:
    --- statistics : dict where the keys are ['avg_nodes', 'avg_edges', 'avg_degree', 'label']
    """

    nodes_per_graph = []
    edges_per_graph = []
    labels = []

    for split, graphs in dataset.items():
        for graph in graphs:
            nodes_per_graph.append(graph.num_nodes)
            edges_per_graph.append(graph.num_edges)
            labels.append(int(graph.y.item()))

    nodes_per_graph = torch.tensor(nodes_per_graph, dtype=torch.float)
    edges_per_graph = torch.tensor(edges_per_graph, dtype=torch.float)

    # Average raw in-degree and out-degree for each directed graph:
    # mathematically, avg_in_degree = avg_out_degree = |E| / |V|
    avg_degree_per_graph = edges_per_graph / nodes_per_graph

    statistics = {
          "avg_nodes": nodes_per_graph.mean().item(),
          "avg_edges": edges_per_graph.mean().item(),
          "avg_degree": avg_degree_per_graph.mean().item(),  #calculates avg to each graph, and then avg across all the graphs
          "label": torch.unique(torch.tensor(labels)).tolist()
    }
    return statistics




### Subsection 3: Shuffle lists ###
# In the Out-of-Distribution experiment, we split the "Distinct" dataset to train, val and future_train and test.
# Therefore, we want to randomly shuffle the lists in advance. Moreover, we shuffle the Common and Original datasets as well.

def shuffle_dict(datasets_name, datasets):
    """
    Shuffles PyG datasets stored in a one-level or two-level dictionary.
    The dictionary is modified in place.

    Inputs:
    --- datasets_name: string
    --- datasets : dict where the values may be dict themselves or PyG datasets.
    Output: None
    """

    print(f"Shuffles {datasets_name} ...", end="")
    for key, value in datasets.items():
        # Checks if the value is dict itself:
        if isinstance(value, dict):
            shuffle_dict_values(value)

        else: # value is PyG dataset
            datasets[key] = value.shuffle()

    print("Done!")
    return None # Changes are in-place


def shuffle_dict_values(datasets):
    """
    Recursively shuffles every PyG dataset stored inside a nested dictionary.
    The dictionary is modified in place.
    """

    for key, value in datasets.items():
        if isinstance(value, dict):
            shuffle_dict_values(value)
        else:
            datasets[key] = value.shuffle()

    return None # Changes are in-place




### Subsection 4: Pandas Dataframe ###
# Our baselines and the main experiments include running classic ML and MLP models, which expect for pandas dataframe.
# We aimed that the features in the dataframe be as much as possible identical to what the graphs themselves embody.
# Implemention has one main invariant: inner-order is kept, namely asking for the first 10 graphs from dict with
# specifc malware_type, is the same as to slice the first 10 rows from the dataframe of the specific maleare_type.


# Global variables:
# The feature columns' names for the dataframes
metadata_features = ["dataset_name", "malware_type", "graph_index", "official_split"]
statistical_features = [
    "log1p_num_nodes",
    "log1p_num_edges",
    "avg_log1p_in_degree",
    "avg_log1p_out_degree",
    "std_log1p_in_degree",
    "std_log1p_out_degree",
    "max_log1p_in_degree",
    "max_log1p_out_degree",
    "frac_isolated_nodes"
    ]
label_feature = ["label"]
features = metadata_features + statistical_features + label_feature

def convert_to_dataframe(malnet_datasets):
    """
    Convert the MalNet datasets dictionary into a graph-level DataFrame.
    Expected structure:
    malnet_datasets[dataset_name][malware_type][split] -> ordered collection of graphs
    Each graph becomes exactly one row in the DataFrame.

    Invariant:
    --- The original graph order inside every dataset/malware subset is preserved.
    --- ggraph_index is the graph's index inside its malware type and split.
    Input:
    --- malenet_datasets : dict where the values may be dict themselves or PyG datasets.
    Output:
    --- df : pandas DataFrame
    """

    split_id_to_name = {0: "train", 1: "val", 2: "test"}
    rows = []

    for dataset_name, malware_types in malnet_datasets.items():

        for malware_type, splits in malware_types.items():

            for split_name, graphs in splits.items():

              # enumerate preserves the original order inside each split
              for graph_index, graph in enumerate(graphs):

                # Basic validation
                if graph.x is None:
                    raise ValueError(
                        f"Graph {graph_index} in "
                        f"{dataset_name}/{malware_type} has no node features (graph.x)."
                    )

                if graph.x.ndim != 2 or graph.x.shape[1] < 2:
                    raise ValueError(
                        f"Graph {graph_index} in "
                        f"{dataset_name}/{malware_type} must contain at least "
                        f"two node features: log1p(in-degree) and log1p(out-degree)."
                    )

                # For the official_split feature
                split_id = int(graph.official_split.item())
                official_split = split_id_to_name[split_id]

                if official_split != split_name:
                    raise ValueError(
                        f"Graph {graph_index} in "
                        f"{dataset_name}/{malware_type}/{split_name} has "
                        f"official_split={official_split}."
                    )

                # Node degree features already stored in log1p scale
                in_degree = graph.x[:, 0]
                out_degree = graph.x[:, 1]

                # Avoid NaN std for a graph containing only one node
                if graph.num_nodes > 1:
                    std_in_degree = in_degree.std().item()
                    std_out_degree = out_degree.std().item()
                else:
                    std_in_degree = 0.0
                    std_out_degree = 0.0

                # Fraction of isolated nodes
                # log1p(0) = 0, so isolation can be detected directly from graph.x
                isolated_nodes = (in_degree == 0) & (out_degree == 0)
                frac_isolated_nodes = isolated_nodes.float().mean().item()

                row = {
                    # Metadata / graph identification
                    "dataset_name": str(dataset_name),
                    "malware_type": str(malware_type),
                    "graph_index": graph_index,
                    "official_split": official_split,

                    # Basic graph statistics
                    "log1p_num_nodes": np.log1p(graph.num_nodes),
                    "log1p_num_edges": np.log1p(graph.num_edges),

                    # Statistics over the log1p degree node features
                    "avg_log1p_in_degree": in_degree.mean().item(),
                    "avg_log1p_out_degree": out_degree.mean().item(),

                    "std_log1p_in_degree": std_in_degree,
                    "std_log1p_out_degree": std_out_degree,

                    "max_log1p_in_degree": in_degree.max().item(),
                    "max_log1p_out_degree": out_degree.max().item(),

                    "frac_isolated_nodes": frac_isolated_nodes,

                    # Graph label
                    "label": int(graph.y.item())
                }

                rows.append(row)

    df = pd.DataFrame(rows, columns=features)
    return df



## Section : Overall ###
def main_data_preprocessing():
    """
    Runs the full preprocessing pipeline:
    Loads the datasets, adds log-transformed degree features and binary labels,
    validates the dataset structure, shuffles all dataset subsets,
    creates the DataFrame and statistics, and saves all results.

    Inputs: None
    Outputs:
    --- MalNet_datasets : nested dict in the form
        {dataset_name: {malware_type: {split: PyG dataset_subset}}}.
    --- Malnet_Original_splits : dict where the keys are ['train', 'val', 'test'].
    --- MalNet_datasets_df : pandas DataFrame containing graph-level features.
    --- datasets_statistics : nested dict containing statistics for every malware type.
    """

    # Loads datasets:
    print("\n=== Loading MalNet datasets ===")
    MalNet_datasets, Malnet_Original_splits = load_MalNet_datasets()

    # Validates the nested dataset structure:
    print("\n=== Validating MalNet datasets structure ===")
    print("Validating ...", end="")
    validate_MalNet_datasets_structure(MalNet_datasets)
    print("Done!")

    # Shuffles dictionaries:
    print("\n=== Shuffling (PyG) Datasets === ")
    shuffle_dict("MalNet_datasets", MalNet_datasets)
    shuffle_dict("Malnet_Original_splits", Malnet_Original_splits)

    # Converts the PyG datasets to a Pandas DataFrame:
    print("\n=== Converting MalNet datasets to dataframe ===")
    print("Converting ... ", end="")
    MalNet_datasets_df = convert_to_dataframe(MalNet_datasets)
    print("Done!")

    # Calculates statistics for each malware type:
    datasets_statistics = {}

    print("\n=== Shows Statistics For Each Malware Type ===")
    for dataset_name, dataset in MalNet_datasets.items():

        print(f"\n{dataset_name}:")
        datasets_statistics[dataset_name] = {}

        for malware_type, subset in dataset.items():
            statistics = malware_graph_statistics(subset)
            datasets_statistics[dataset_name][malware_type] = statistics
            # Prints:
            print(f"\n  {malware_type}:")
            print("    avg_nodes:", statistics["avg_nodes"])
            print("    avg_edges:", statistics["avg_edges"])
            print("    avg_degree:", statistics["avg_degree"])
            print("    label:", statistics["label"])


    # Defines the paths of the four output files:
    malnet_datasets_path = os.path.join(DATASETS_PATH, "MalNet_datasets.pt")
    malnet_Original_splits_path = os.path.join(DATASETS_PATH, "Malnet_Original_splits.pt")
    malnet_datasets_df_path = os.path.join(DATASETS_PATH, "MalNet_datasets_df.csv")
    statistics_path = os.path.join(DATASETS_PATH, "datasets_statistics.json")

    # Saves shuffled PyG datasets:
    print("\n=== Saving results ===")
    print("Saving MalNet_datasets ...", end="")
    torch.save(MalNet_datasets, malnet_datasets_path)
    print("Done!")

    print("Saving Malnet_Original_splits ...", end="")
    torch.save(Malnet_Original_splits, malnet_Original_splits_path)
    print("Done!")

    # Saves the Pandas DataFrame:
    print("Saving MalNet_datasets_df ...", end="")
    MalNet_datasets_df.to_csv(malnet_datasets_df_path, index=False)
    print("Done!")

    # Saves statistics results:
    print("Saving statistics results ...", end="")
    with open(statistics_path, "w", encoding="utf-8") as f:
        json.dump(datasets_statistics, f, indent=4)
    print("Done!")

    # Prints the locations of the four saved files:
    print("\n=== Saved Files Locations ===")
    print("MalNet datasets:", malnet_datasets_path)
    print("Original splits:", malnet_Original_splits_path)
    print("Dataframe:", malnet_datasets_df_path)
    print("Statistics:", statistics_path)

    print("\nDone!")
    return (
        MalNet_datasets,
        Malnet_Original_splits,
        MalNet_datasets_df,
        datasets_statistics
    )



if __name__ == "__main__":
    main_data_preprocessing()
