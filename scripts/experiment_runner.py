# The shared runner every experiment in the project is built on.
#
# experiment() receives its training and test sets rather than building them, so the
# main out-of-distribution sweep, the two baselines and the size-generalization
# control can all reuse one loop body while each constructs the sets its own question
# needs. The only set it still derives itself is the incremental pool, because that is
# what a k sweep means: Distinct[zeroDay_type][train][:k].

# General libraries
import csv
import os
import re
import time
from pathlib import Path

# PyTorch:
import torch
from torch_geometric.loader import DataLoader

# Project:
from GNNs_models import GNN_Model, evaluate_GNN_model
from preprocessed_data import get_distinct
from training import (
    RANDOM_STATE,
    build_loss_function,
    set_random_seed,
    train_model,
)


### Global variables ###

# Project paths
SCRIPT_PATH = Path(__file__).resolve().parent
PROJECT_PATH = SCRIPT_PATH.parent
RESULTS_PATH = PROJECT_PATH / "results"
FIGURES_PATH = RESULTS_PATH / "figures"

# The architectures every experiment compares. Not a parameter of experiment(): all
# four are always run, and one series per architecture is what the figures show.
# GT is the transformer-style architecture of the group, taking the place GPS used
# to hold: it keeps attention but computes it over each node's incoming edges, so it
# stays linear in |E| and runs at the same batch size as the rest.
GNN_TYPES = ("GCN", "GIN", "GAT", "GT")

# How many graphs of the zero-day family are moved into the training set.
# Every family has 700 graphs in its train split, which is the 70% the work plan
# describes. The grid is denser at the low end because the question is *when* the
# curve stabilises, and learning curves usually flatten early: evenly spaced points
# spend most of the compute on the part where nothing happens any more.
K_VALUES = (0, 25, 50, 100, 175, 250, 350, 500, 700)

# Model architecture. input_dim is 2 because the node features are
# log1p(in-degree) and log1p(out-degree).
CONFIGURATION = {
    "num_layers": 5,
    "hidden_dim": 64,
    "dropout": 0.5,
    "heads": 4,
    "input_dim": 2,
    "output_dim": 1,
}

# Optimisation
TRAINING_CONFIGURATION = {
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "epochs": 30,
    "batch_size": 32,
}

# Per-architecture batch sizes. All four are linear in |E| - GT included, since its
# attention is over each node's incoming edges rather than globally over the nodes -
# so these all match TRAINING_CONFIGURATION["batch_size"]. The table stays so that an
# architecture whose memory grows with |V| rather than |E| can be given a smaller
# batch without touching the loop that builds the loaders.
BATCH_SIZE_BY_GNN_TYPE = {
    "GCN": 32,
    "GIN": 32,
    "GAT": 32,
    "GT": 32,
}

# One fixed colour and marker per architecture, looked up by name rather than by
# position. A positional palette would hand the first architecture of a partial run
# the colour that GCN owns everywhere else, and the figures would stop being
# comparable side by side. The marker shapes keep the series apart in greyscale, in
# print and for colour-vision deficiency.
STYLE_BY_GNN = {
    "GCN": ("#2a78d6", "o"),
    "GIN": ("#eb6834", "s"),
    "GAT": ("#1baf7a", "^"),
    "GT": ("#eda100", "D"),
}

# The columns written to the results CSV. Only what the figures consume, plus enough
# bookkeeping to tell two runs apart.
RESULT_FIELDS = [
    "gnn_type",
    "zero_day_type",
    "k",
    "num_train_graphs",
    "num_test_graphs",
    "pos_weight",
    "test_binary_f1",
    "test_macro_f1",
    "zero_day_accuracy",
    "train_loss",
    "seconds",
]


### Subsection 1: Model construction ###

def build_model(GNN_type, device):
    """
    Builds a GNN_Model from CONFIGURATION.

    CONFIGURATION also has to be validated here: the attention layers split
    hidden_dim across the heads, so hidden_dim must divide evenly by heads or the
    failure surfaces much later as a shape error inside the attention layer.

    Inputs:
    --- GNN_type: string, one of "GCN", "GIN", "GAT", "GT".
    --- device: torch.device
    Output:
    --- model: GNN_Model on the given device.
    """

    if GNN_type.upper() in {"GAT", "GT"}:
        heads = CONFIGURATION.get("heads")

        if not heads or CONFIGURATION["hidden_dim"] % heads != 0:
            raise ValueError(
                f"{GNN_type} needs hidden_dim to be divisible by heads, got "
                f"hidden_dim={CONFIGURATION['hidden_dim']} and heads={heads}."
            )

    model = GNN_Model(GNN_type, **CONFIGURATION)

    return model.to(device)


def build_optimizer(model):
    """ Builds the optimiser named in TRAINING_CONFIGURATION. """
    learning_rate = TRAINING_CONFIGURATION["learning_rate"]
    return torch.optim.Adam(model.parameters(), lr=learning_rate)

def batch_size_for(GNN_type):
    """ Returns the batch size for one architecture, honouring BATCH_SIZE_BY_GNN_TYPE. """
    return BATCH_SIZE_BY_GNN_TYPE.get(
        GNN_type.upper(), TRAINING_CONFIGURATION["batch_size"]
    )


### Subsection 2: Results storage ###

def slug(label):
    """
    Turns a label into a file name. Keeps the characters a malware type actually uses,
    so "clicker++trojan" stays readable, and replaces anything else, so a label a
    future experiment invents is still a legal file name on every platform.

    Input:
    --- label: string
    Output:
    --- slugged: string
    """

    return re.sub(r"[^A-Za-z0-9._+-]", "_", str(label))


def results_file_path(experiment_name, zeroDay_type, results_dir=None):
    """
    Returns the CSV path holding one zero-day family's results.

    One experiment() call covers every architecture for a single family, so one call
    is one file: no two calls ever append to the same CSV, and two families running
    as separate jobs cannot collide.

    Inputs:
    --- experiment_name: string, names the sub-directory under results/.
    --- zeroDay_type: string, names the file.
    --- results_dir: Path or string overriding results/<experiment_name>.
    Output:
    --- path: Path
    """

    results_dir = Path(
        RESULTS_PATH / slug(experiment_name) if results_dir is None else results_dir
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    return results_dir / f"{slug(zeroDay_type)}.csv"


def load_results(path):
    """
    Reads a results CSV into a dict keyed by (gnn_type, k).
    A sweep is hundreds of trainings, so an interrupted run has to be resumable.

    Input:
    --- path: Path
    Output:
    --- rows: dict mapping (gnn_type, int k) to the row dict.
    """

    if not os.path.isfile(path):
        return {}

    rows = {}

    with open(path, "r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            rows[(row["gnn_type"], int(row["k"]))] = row

    return rows


def append_result(path, row):
    """
    Appends one finished run to the results CSV, writing the header first if the
    file is new. Flushed immediately so an interrupted sweep loses at most one run.

    Inputs:
    --- path: Path
    --- row: dict with the keys of RESULT_FIELDS.
    Output: None
    """

    write_header = not os.path.isfile(path)

    with open(path, "a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_FIELDS)

        if write_header:
            writer.writeheader()

        writer.writerow(row)

    return None


def matches_current_sets(row, num_train_graphs, num_test_graphs):
    """
    Says whether a stored row was produced from the sets this call is running.

    A matching (gnn_type, k) key used to be proof, back when experiment() built its
    own sets and they could not differ between runs. Now the caller supplies them, so
    a driver whose base set changed would otherwise replay stale numbers under a key
    that still matches. Both size columns are already in the CSV, so the check costs
    no new fields.

    Inputs:
    --- row: dict read from the CSV.
    --- num_train_graphs: int, the size this call would train on.
    --- num_test_graphs: int, the size this call would test on.
    Output:
    --- matches: bool
    """

    return (
        int(row["num_train_graphs"]) == num_train_graphs
        and int(row["num_test_graphs"]) == num_test_graphs
    )


### Subsection 3: The experiment ###

def zero_day_train_slice(zeroDay_type, k):
    """
    The first k graphs of one Distinct family's train split, which is the pool the k
    sweep draws from.

    The split was shuffled during preprocessing, so [:k] is a random sample; and
    because the slices are nested, every k contains all the graphs of the smaller
    ones. That is what makes the sequence a learning curve rather than a set of
    unrelated runs.

    Distinct is only looked up when k > 0. That is what lets an experiment with no
    single zero-day family pass a label of its own with K_values=(0,) and never have
    that label used as a key.

    Inputs:
    --- zeroDay_type: string, a malware type of the Distinct dataset.
    --- k: int
    Output:
    --- graphs: list of PyG Data objects, empty when k is 0.
    """

    if k <= 0 or not zeroDay_type:
        return []

    return list(get_distinct()[zeroDay_type]["train"][:k])


def experiment(
    zeroDay_type,
    base_train_set,
    base_test_set,
    zeroDay_for_test,
    experiment_name,
    K_values=K_VALUES,
    device=None,
    results_dir=None,
    resume=True
):
    """
    Runs one incremental experiment: every architecture, every k, one zero-day family.

    For each architecture and each k it trains on
        base_train_set + Distinct[zeroDay_type][train][:k]
    and tests on
        base_test_set + zeroDay_for_test
    then additionally measures accuracy on zeroDay_for_test alone, which is what shows
    how well the newly seen family itself is recognised.

    The caller supplies the sets, so what the experiment *is* lives in the caller and
    only the sweep lives here. Those sets hold the very graph objects the cached
    datasets hold; sharing them across architectures and k values is safe, and is what
    keeps a whole sweep in memory, but it means no caller may modify a graph in place.
    graph.x is a view into the collated dataset every other experiment reads.

    Inputs:
    --- zeroDay_type: string. The family whose train split feeds the k sweep, and the
        name the CSV and the figure are filed under.
    --- base_train_set: list of PyG Data objects, the part of training that is fixed.
    --- base_test_set: list of PyG Data objects, the part of the test set that is not
        the zero-day family.
    --- zeroDay_for_test: list of PyG Data objects held out from the zero-day family.
        May be empty, for an experiment that isolates no sub-population; the zero-day
        accuracy is then NaN.
    --- experiment_name: string, the sub-directory of results/ this run writes to.
    --- K_values: iterable of ints.
    --- device: torch.device, defaults to cuda when available.
    --- results_dir: Path or string for the CSV, defaults to results/<experiment_name>.
    --- resume: bool, skip (architecture, k) pairs already present in the CSV.
    Outputs:
    --- F1_results: dict mapping architecture -> list of (k, macro F1 on the whole
        test set)
    --- Accuracy_for_zeroDay: dict mapping architecture -> list of (k, accuracy on the
        zero-day graphs alone)
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path = results_file_path(experiment_name, zeroDay_type, results_dir)
    finished_runs = load_results(path) if resume else {}

    base_train_set = list(base_train_set)

    # The held-out zero-day graphs are identical at every k, so the curve measures the
    # effect of k and nothing else.
    #
    # No shuffle, and concatenating the parts in a fixed order biases nothing: the
    # loaders below are built with shuffle=False, evaluate_GNN_model accumulates every
    # prediction before it computes a metric, and every metric it returns is
    # order-invariant. Batch composition cannot change a prediction either, because
    # model.eval() puts BatchNorm on its running statistics. A fixed order is simply
    # the reproducible choice.
    zeroDay_for_test = list(zeroDay_for_test)
    test_set = list(base_test_set) + zeroDay_for_test

    print(f"\n=== {experiment_name}: {zeroDay_type} ===")
    print(
        f"base train={len(base_train_set)}, test={len(test_set)}, "
        f"held-out zero-day={len(zeroDay_for_test)}"
    )

    F1_results = {}
    Accuracy_for_zeroDay = {}

    for GNN_type in GNN_TYPES:
        F1_results[GNN_type] = []
        Accuracy_for_zeroDay[GNN_type] = []

        # Only the batch size differs between architectures, so the evaluation sets
        # are the same lists rebuilt into per-architecture loaders.
        batch_size = batch_size_for(GNN_type)
        test_loader = DataLoader(test_set, batch_size, shuffle=False)

        # An experiment that isolates no sub-population passes an empty list here.
        # evaluate_GNN_model raises on an empty loader, so there is nothing to build.
        zero_day_loader = (
            DataLoader(zeroDay_for_test, batch_size, shuffle=False)
            if zeroDay_for_test else None
        )

        print(f"\n-- {GNN_type.upper()}  (batch size {batch_size})")

        for k in K_values:
            # Step 3.2.1 - training set. list() is load-bearing: adding a PyG subset
            # to a list is a TypeError.
            train_set = base_train_set + zero_day_train_slice(zeroDay_type, k)

            # Resume: reuse a run that already finished, but only if it was produced
            # from the sets this call is holding.
            stored_run = finished_runs.get((GNN_type, k))

            if stored_run is not None:

                if matches_current_sets(stored_run, len(train_set), len(test_set)):
                    F1_results[GNN_type].append(
                        (k, float(stored_run["test_macro_f1"]))
                    )
                    Accuracy_for_zeroDay[GNN_type].append(
                        (k, float(stored_run["zero_day_accuracy"]))
                    )
                    print(f"   k={k:4d}  (already in {path.name}, skipped)")
                    continue

                print(
                    f"   k={k:4d}  the stored run used "
                    f"{stored_run['num_train_graphs']}/{stored_run['num_test_graphs']}"
                    f" graphs but this call has {len(train_set)}/{len(test_set)},"
                    " recomputing"
                )

            started_at = time.time()

            # The same seed for every point. Each run builds a fresh model, so this
            # starts them all from an identical initialisation and the only thing
            # that changes along a curve is k. Run order cannot shape the result,
            # and a resumed sweep reproduces the points it already wrote.
            set_random_seed(RANDOM_STATE)

            train_loader = DataLoader(train_set, batch_size, shuffle=True)

            # Step 3.2.3 - loss, rebuilt for this k because pos_weight depends on it
            loss_function, pos_weight = build_loss_function(train_set, device)

            # Step 3.2.4 and 3.2.5 - a fresh model per point, then training
            gnn_model = build_model(GNN_type, device)
            optimizer = build_optimizer(gnn_model)

            history = train_model(
                gnn_model,
                device,
                train_loader,
                loss_function,
                optimizer,
                epochs=TRAINING_CONFIGURATION["epochs"],
            )

            # Step 3.2.6 - metrics over the whole test set
            test_results = evaluate_GNN_model(
                gnn_model, device, test_loader, loss_function
            )

            # Step 3.2.7 - accuracy over the zero-day graphs alone. Every label there
            # is 1, so accuracy equals recall on that family and ROC AUC is undefined
            # (evaluate_GNN_model returns NaN, which is expected here).
            zero_day_accuracy = (
                evaluate_GNN_model(
                    gnn_model, device, zero_day_loader, loss_function
                )["accuracy"]
                if zero_day_loader is not None else float("nan")
            )

            seconds = time.time() - started_at

            # Steps 3.2.8 and 3.2.9
            F1_results[GNN_type].append((k, test_results["macro_f1"]))
            Accuracy_for_zeroDay[GNN_type].append((k, zero_day_accuracy))

            append_result(path, {
                "gnn_type": GNN_type.upper(),
                "zero_day_type": zeroDay_type,
                "k": k,
                "num_train_graphs": len(train_set),
                "num_test_graphs": len(test_set),
                "pos_weight": f"{pos_weight:.6f}",
                "test_binary_f1": f"{test_results['binary_f1']:.6f}",
                "test_macro_f1": f"{test_results['macro_f1']:.6f}",
                "zero_day_accuracy": f"{zero_day_accuracy:.6f}",
                "train_loss": f"{history['train_losses'][-1]:.6f}",
                "seconds": f"{seconds:.1f}",
            })

            print(
                f"   k={k:4d}  train={len(train_set):5d}  pos_weight={pos_weight:.3f}"
                f"  macro_F1={test_results['macro_f1']:.4f}"
                f"  zero_day_acc={zero_day_accuracy:.4f}"
                f"  ({seconds:.0f}s)"
            )

    print(f"\nResults written to {path}")

    return F1_results, Accuracy_for_zeroDay


### Subsection 4: Plots ###

def plot_experiment_results(
    zeroDay_type,
    K_values,
    F1_results,
    Accuracy_results,
    experiment_name="Out_of_Distribution",
    output_dir=None,
    ylim=(0.0, 1.02)
):
    """
    Draws the two learning curves of one zero-day family, side by side: macro F1 over
    the whole test set, and accuracy over the zero-day family alone. One line per
    architecture, sharing an x axis of k and a y axis of 0 to 1.

    Both panels use the same fixed y range so the curves can be compared directly
    and a flat curve looks flat. Pass ylim=None to auto-scale instead.

    Inputs:
    --- zeroDay_type: string
    --- K_values: iterable of ints, the x axis
    --- F1_results: dict mapping architecture -> list of (k, macro F1)
    --- Accuracy_results: dict mapping architecture -> list of (k, accuracy)
    --- experiment_name: string, part of the file name
    --- output_dir: Path or string, defaults to results/figures
    --- ylim: (low, high) tuple or None
    Output:
    --- saved_paths: list of Path, the files written
    """

    K_values = list(K_values)

    # A single k is a point, not a curve. An experiment that sweeps nothing still gets
    # its CSV; it just gets no figure.
    if len(K_values) < 2:
        print("Nothing to plot: a learning curve needs at least two k values.")
        return []

    import matplotlib
    matplotlib.use("Agg")  # No display on a compute node
    import matplotlib.pyplot as plt

    output_dir = Path(FIGURES_PATH if output_dir is None else output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    surface = "#fcfcfb"
    ink = "#0b0b0b"
    ink_secondary = "#52514e"
    grid_color = "#dedcd7"

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.facecolor": surface,
        "axes.facecolor": surface,
        "savefig.facecolor": surface,
        "text.color": ink,
        "axes.labelcolor": ink_secondary,
        "xtick.color": ink_secondary,
        "ytick.color": ink_secondary,
    })

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)

    panels = [
        (axes[0], F1_results, "Macro F1 on the full test set",
         "the whole test set"),
        (axes[1], Accuracy_results, f"Accuracy on {zeroDay_type} only",
         "held-out graphs of that family"),
    ]

    # GNN_TYPES rather than the dictionary's own order, so an architecture keeps the
    # same place in the legend of every figure.
    series = [GNN_type for GNN_type in GNN_TYPES if GNN_type in F1_results]

    for axis, results, title, subtitle in panels:

        for GNN_type in series:
            points = sorted(results.get(GNN_type, []))

            if not points:
                continue

            color, marker = STYLE_BY_GNN[GNN_type]
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]

            axis.plot(
                xs, ys,
                color=color,
                marker=marker,
                markersize=5.5,
                markeredgecolor=surface,
                markeredgewidth=0.8,
                linewidth=2.0,
                label=GNN_type,
                zorder=3,
            )

        axis.set_title(title, color=ink, pad=10, loc="left")
        axis.set_xlabel(f"k  -  zero-day graphs added to training\n{subtitle}")
        # The k grid is deliberately dense at the low end, where the curve actually
        # moves, so the first few labels collide when they are drawn flat.
        axis.set_xticks(K_values)
        axis.tick_params(axis="x", labelsize=8.5)

        for label in axis.get_xticklabels():
            label.set_rotation(45)
            label.set_horizontalalignment("right")
        axis.grid(axis="y", color=grid_color, linewidth=0.8, zorder=0)
        axis.set_axisbelow(True)

        for side in ("top", "right"):
            axis.spines[side].set_visible(False)

        for side in ("left", "bottom"):
            axis.spines[side].set_color(grid_color)

        if ylim is not None:
            axis.set_ylim(*ylim)

    axes[0].set_ylabel("score")

    figure.suptitle(
        f"{zeroDay_type} - incremental out-of-distribution learning curves",
        x=0.012, ha="left", fontsize=13, color=ink,
    )

    # One legend for both panels. With four series direct labels would fit, but both
    # panels carry the same series, so labelling them once below the pair is what
    # keeps them readable as a unit. The marker shapes carry identity as well.
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles, labels,
        loc="lower center", ncol=len(labels), frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    figure.tight_layout(rect=(0, 0.06, 1, 0.94))

    saved_paths = []
    file_name = f"{slug(experiment_name)}_{slug(zeroDay_type)}"

    for suffix in ("png", "pdf"):
        target = output_dir / f"{file_name}.{suffix}"
        figure.savefig(target, dpi=200, bbox_inches="tight")
        saved_paths.append(target)

    plt.close(figure)

    print("Figures written to:")
    for target in saved_paths:
        print(f"--- {target}")

    return saved_paths
