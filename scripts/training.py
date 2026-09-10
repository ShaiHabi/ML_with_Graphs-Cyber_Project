# Reusable training utilities for every experiment in the project.
# Nothing here is specific to the main OOD experiment: baselines #1 and #2 and the
# size-generalization control are meant to call the same functions.


# General libraries
import copy
import random
import time

# General
import numpy as np

# PyTorch:
import torch

# Project:
from GNNs_models import train_one_epoch_GNN_model, evaluate_GNN_model


# Global variables:
RANDOM_STATE = 42

# The order splits are concatenated in, so that a graph list is always built the
# same way regardless of the insertion order of the dictionaries.
SPLIT_ORDER = ("train", "val", "test")


def set_random_seed(seed):
    """
    Sets the random seed for reproducible experiments.
    Identical to the function in main.py; duplicated here so that the scripts
    package does not have to import from the project root.

    Input:
    --- seed: int
    Output: None
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return None


def collect_graphs(dataset_by_type, malware_types=None, splits=None):
    """
    Flattens one dataset of the {malware_type: {split: PyG dataset_subset}} form
    into a single ordered list of graphs.

    The malware types are visited in sorted order and the splits in train, val,
    test order, so the same arguments always produce the same list.

    Inputs:
    --- dataset_by_type: dict, for example MalNet_datasets["Original"].
    --- malware_types: iterable of type names to keep, or None for all of them.
    --- splits: iterable of split names to keep, or None for all of them.
    Output:
    --- graphs: list of PyG Data objects.
    """

    if malware_types is not None:
        malware_types = set(malware_types)

    if splits is not None:
        splits = set(splits)

    graphs = []

    for malware_type in sorted(dataset_by_type.keys()):

        if malware_types is not None and malware_type not in malware_types:
            continue

        splits_of_type = dataset_by_type[malware_type]

        for split in SPLIT_ORDER:

            if split not in splits_of_type:
                continue

            if splits is not None and split not in splits:
                continue

            graphs.extend(list(splits_of_type[split]))

    # No extra shuffle here, and the ordering by type then split introduces no bias.
    # A training DataLoader is always built with shuffle=True, so it reshuffles the
    # indices every epoch and the order of this list never reaches a batch. An
    # evaluation DataLoader uses shuffle=False, but evaluate_GNN_model accumulates
    # every prediction before it computes a metric, and all of those metrics are
    # order-invariant. Leaving the order deterministic is what lets a resumed sweep
    # rebuild exactly the same training set it used the first time.
    return graphs


def label_counts(graphs):
    """
    Counts benign (label 0) and malicious (label 1) graphs.

    Input:
    --- graphs: list of PyG Data objects.
    Output:
    --- (num_negative, num_positive): tuple of ints
    """

    num_positive = 0

    for graph in graphs:
        num_positive += int(graph.y.item())

    return len(graphs) - num_positive, num_positive


def build_loss_function(train_graphs, device):
    """
    Builds BCEWithLogitsLoss with pos_weight computed from the given training set
    only, as the work plan requires. Must be rebuilt whenever the training set
    changes, because in the main experiment pos_weight depends on k.

    Direction of the correction: BCEWithLogitsLoss multiplies the *positive* term
    of the loss by pos_weight, so the weight that balances the two classes is
    num_negative / num_positive. In this project malware (1) outnumbers benign (0)
    roughly four to one, so pos_weight comes out around 0.25 - it damps the
    over-represented class.

    A training set with no positives at all is allowed, and gets pos_weight 1. That is
    the k = 0 point of Per_Family_Analysis, which trains on benign graphs only. The
    weight is arbitrary there and cannot be anything else: it scales a term of the loss
    that has no examples to apply it to.

    Inputs:
    --- train_graphs: list of PyG Data objects.
    --- device: torch.device
    Outputs:
    --- loss_fn: torch.nn.BCEWithLogitsLoss
    --- pos_weight: float, the same weight as a plain number, so that callers can
        record it without counting the labels a second time.
    """

    num_negative, num_positive = label_counts(train_graphs)

    # An empty negative class is fatal and, worse, silent: num_negative / num_positive
    # is then 0, which multiplies the whole positive term of the loss by zero, and the
    # model quietly learns to answer 0 to everything while its loss curve looks
    # healthy. Nothing in this project is supposed to train on malware alone, so this
    # can only be a set built wrong.
    if num_negative == 0:
        raise ValueError(
            "A training set of nothing but malicious graphs cannot be weighted: "
            f"got {num_negative} benign and {num_positive} malicious graphs."
        )

    # The other direction is deliberate. Per_Family_Analysis starts its curve at k = 0,
    # where the training set is benign graphs only, and there the ratio is undefined
    # rather than wrong: no positive example exists for pos_weight to scale. 1 is the
    # neutral choice and leaves the loss exactly the unweighted BCE it already is.
    pos_weight = num_negative / num_positive if num_positive else 1.0

    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [pos_weight],
            dtype=torch.float32,
            device=device
        )
    )

    return loss_fn, pos_weight


def train_model(
    model,
    device,
    train_loader,
    loss_fn,
    optimizer,
    epochs,
    val_loader=None,
    selection_metric="macro_f1",
    log_every=0
):
    """
    Trains a model for several epochs. This is the loop the project was missing:
    GNNs_models only provides a single epoch.

    If val_loader is given, the epoch with the best selection_metric on it is kept
    and its weights are restored before returning. If it is None the model is simply
    trained for the full number of epochs, which is the configuration the main
    experiment uses (its whole Original dataset goes into training, so no split is
    left over for validation).

    Inputs:
    --- model: torch.nn.Module
    --- device: torch.device
    --- train_loader: torch_geometric.loader.DataLoader
    --- loss_fn: torch.nn.Module
    --- optimizer: torch.optim.Optimizer
    --- epochs: int
    --- val_loader: torch_geometric.loader.DataLoader or None
    --- selection_metric: string, a key of the dict evaluate_GNN_model returns
    --- log_every: int, print an epoch line every log_every epochs (0 = never)
    Output:
    --- history: dict with the per-epoch training losses, the per-epoch validation
        results, the index of the selected epoch and the elapsed seconds.
    """

    train_losses = []
    validation_results = []

    best_score = None
    best_epoch = None
    best_state = None

    started_at = time.time()

    for epoch in range(1, epochs + 1):

        train_loss = train_one_epoch_GNN_model(
            model, device, train_loader, optimizer, loss_fn
        )
        train_losses.append(train_loss)

        epoch_line = f"    epoch {epoch:3d}/{epochs}  train_loss={train_loss:.4f}"

        if val_loader is not None:
            results = evaluate_GNN_model(model, device, val_loader, loss_fn)
            validation_results.append(results)

            score = results[selection_metric]
            epoch_line += f"  val_{selection_metric}={score:.4f}"

            if best_score is None or score > best_score:
                best_score = score
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                epoch_line += "  *"

        if log_every and (epoch % log_every == 0 or epoch == epochs):
            print(epoch_line)

    # Restores the best epoch's weights when a validation set was used
    if best_state is not None:
        model.load_state_dict(best_state)

    history = {
        "train_losses": train_losses,
        "validation_results": validation_results,
        "selected_epoch": best_epoch if best_epoch is not None else epochs,
        "selection_metric": selection_metric if val_loader is not None else None,
        "seconds": time.time() - started_at
    }

    return history
