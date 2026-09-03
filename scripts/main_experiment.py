# The main experiment: Out-of-Distribution and Learning Analysis.
#
# Training: Original + Distinct[zeroDay_type][train][:k]
# Validation: None (the held-out graphs of the zero-day family are in the test set due to lack of data)
# Test: Distinct[zeroDay_type][validation] + Distinct[zeroDay_type][test]

# General libraries
import csv
import os
import time
from pathlib import Path

# PyTorch:
import torch
from torch_geometric.loader import DataLoader

# Project:
from GNNs_models import GNN_Model, evaluate_GNN_model
from training import (
    RANDOM_STATE,
    build_loss_function,
    collect_graphs,
    set_random_seed,
    train_model,
)


### Global variables ###

# Project paths
SCRIPT_PATH = Path(__file__).resolve().parent
PROJECT_PATH = SCRIPT_PATH.parent
RESULTS_PATH = PROJECT_PATH / "results"
MAIN_EXPERIMENT_RESULTS_PATH = RESULTS_PATH / "main_experiment"
FIGURES_PATH = RESULTS_PATH / "figures"

# How many graphs of the zero-day family are moved into the training set.
# Every family has 700 graphs in its train split, which is the 70% the work plan
# describes. The grid is denser at the low end because the question is *when* the
# curve stabilises, and learning curves usually flatten early: evenly spaced points
# spend most of the compute on the part where nothing happens any more.
K_VALUES = [0, 25, 50, 100, 175, 250, 350, 500, 700]

# The five malware families of the "Distinct" dataset. It holds no benign graphs.
# The order fixes the colour each family gets in every figure.
ZERODAYS_TYPES = ["clicker++trojan", "malware", "riskware", "spr", "spyware"]

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

# GPSConv runs global attention over the nodes of each graph, so its memory grows
# with the largest graph in the batch, not with the average one. MalNet-Tiny graphs
# reach 14,166 nodes: measured on this codebase, one GPS layer over a batch of 8
# such graphs needs about 4.8 GB, while a batch of 32 needs far more memory than a
# normal machine has. The other three architectures are linear in |E| and are fine
# at 32.
BATCH_SIZE_BY_GNN_TYPE = {
    "GPS": 4,
    "GCN": 32,
    "GIN": 32,
    "GAT": 32,
}

# Categorical palette, one fixed slot per family, plus a marker shape so the series
# stay distinguishable in greyscale, in print and for colour-vision deficiency.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SERIES_MARKERS = ["o", "s", "^", "D", "v"]

# The columns written to the results CSV.
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

    CONFIGURATION also has to be validated here: GPSConv splits hidden_dim across
    the attention heads, so hidden_dim must divide evenly by heads or the failure
    surfaces much later as a shape error inside the attention layer.

    Inputs:
    --- GNN_type: string, one of "GCN", "GIN", "GAT", "GPS".
    --- device: torch.device
    Output:
    --- model: GNN_Model on the given device.
    """

    if GNN_type.upper() in {"GAT", "GPS"}:
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

def results_file_path(GNN_type, results_dir=None):
    """ Returns the CSV path holding one architecture's results. """
    results_dir = Path(
        MAIN_EXPERIMENT_RESULTS_PATH if results_dir is None else results_dir
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    return results_dir / f"{GNN_type.upper()}.csv"


def load_results(path):
    """
    Reads a results CSV into a dict keyed by (zero_day_type, k).
    A sweep is hundreds of trainings, so an interrupted run has to be resumable.

    Input:
    --- path: Path
    Output:
    --- rows: dict mapping (zero_day_type, int k) to the row dict.
    """

    if not os.path.isfile(path):
        return {}

    rows = {}

    with open(path, "r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            rows[(row["zero_day_type"], int(row["k"]))] = row

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


### Subsection 3: The experiment ###

def experiment(
    GNN_type,
    datasets,
    device=None,
    results_dir=None,
    resume=True
):
    """
    Runs the incremental out-of-distribution experiment for one architecture.

    For each zero-day family X and each k:
      train on   Original (all types, all splits) + Distinct[X][train][:k]
      test on    Common (all types, all splits) + Distinct[X][val] + Distinct[X][test]
    and additionally measure accuracy on the zero-day part of the test set alone,
    which is what shows how well the newly seen family itself is recognised.

    The Distinct train split was shuffled during preprocessing, so [:k] is a random
    sample; and because the slices are nested, every k contains all the graphs of
    the smaller ones. That is what makes the sequence a learning curve rather than
    a set of unrelated runs.

    Inputs:
    --- GNN_type: string, one of "GCN", "GIN", "GAT", "GPS".
    --- datasets: the MalNet_datasets dict, with the "Original", "Common" and
        "Distinct" keys, each of the {malware_type: {split: subset}} form.
    --- device: torch.device, defaults to cuda when available.
    --- results_dir: Path or string for the CSV, defaults to results/main_experiment.
    --- resume: bool, skip (family, k) pairs already present in the CSV.
    Outputs:
    --- F1_results: dict mapping family -> list of (k, macro F1 on the whole test set)
    --- Accuracy_for_zeroDay: dict mapping family -> list of (k, accuracy on the
        zero-day graphs only)
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path = results_file_path(GNN_type, results_dir)
    finished_runs = load_results(path) if resume else {}

    batch_size = batch_size_for(GNN_type)

    # Flattens Original and Common datasets because they don't change between runs.
    # Building them once matters: every k below reuses the same lists.
    print(f"\n=== Main experiment: {GNN_type.upper()} ===")
    print("Flattening Original and Common ...", end="", flush=True)
    original_graphs = collect_graphs(datasets["Original"])
    common_graphs = collect_graphs(datasets["Common"])
    print(f"Done! Original={len(original_graphs)}, Common={len(common_graphs)}")

    F1_results = {}
    Accuracy_for_zeroDay = {}

    for zero_day_type in ZERODAYS_TYPES:
        F1_results[zero_day_type] = []
        Accuracy_for_zeroDay[zero_day_type] = []

        family_splits = datasets["Distinct"][zero_day_type]

        # Step 3.1 - the held-out part of the family. It is identical at every k,
        # so the curve measures the effect of k and nothing else.
        #
        # No shuffle, and concatenating val before test biases nothing: the loaders
        # below are built with shuffle=False, evaluate_GNN_model accumulates every
        # prediction before it computes a metric, and every metric it returns is
        # order-invariant. Batch composition cannot change a prediction either,
        # because model.eval() puts BatchNorm on its running statistics. A fixed
        # order is simply the reproducible choice.
        zeroDay_for_test = (
            list(family_splits["val"]) + list(family_splits["test"])
        )
        zero_day_loader = DataLoader(zeroDay_for_test, batch_size, shuffle=False)

        family_train_pool = family_splits["train"]

        print(
            f"\n-- {zero_day_type}: pool={len(family_train_pool)} graphs, "
            f"held out={len(zeroDay_for_test)} graphs"
        )

        for k in K_VALUES:
            # Resume: reuse a run that already finished
            if (zero_day_type, k) in finished_runs:
                row = finished_runs[(zero_day_type, k)]
                F1_results[zero_day_type].append((k, float(row["test_macro_f1"])))
                Accuracy_for_zeroDay[zero_day_type].append(
                    (k, float(row["zero_day_accuracy"]))
                )
                print(f"   k={k:4d}  (already in {path.name}, skipped)")
                continue

            started_at = time.time()

            # The same seed for every point. Each run builds a fresh model, so this
            # starts them all from an identical initialisation and the only thing
            # that changes along a curve is k. Run order cannot shape the result,
            # and a resumed sweep reproduces the points it already wrote.
            set_random_seed(RANDOM_STATE)

            # Step 3.2.1 - training set
            new_family_graphs = list(family_train_pool[:k]) if k > 0 else []
            train_set = original_graphs + new_family_graphs

            # Step 3.2.2 - test set
            test_set = common_graphs + zeroDay_for_test

            train_loader = DataLoader(train_set, batch_size, shuffle=True)
            test_loader = DataLoader(test_set, batch_size, shuffle=False)

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
                val_loader=None,
            )

            # Step 3.2.6 - metrics over the whole test set
            test_results = evaluate_GNN_model(
                gnn_model, device, test_loader, loss_function
            )

            # Step 3.2.7 - accuracy over the zero-day graphs alone. Every label
            # there is 1, so accuracy equals recall on that family and ROC AUC is
            # undefined (evaluate_GNN_model returns NaN, which is expected here).
            zero_day_results = evaluate_GNN_model(
                gnn_model, device, zero_day_loader, loss_function
            )

            seconds = time.time() - started_at

            # Steps 3.2.8 and 3.2.9
            F1_results[zero_day_type].append((k, test_results["macro_f1"]))
            Accuracy_for_zeroDay[zero_day_type].append(
                (k, zero_day_results["accuracy"])
            )

            append_result(path, {
                "gnn_type": GNN_type.upper(),
                "zero_day_type": zero_day_type,
                "k": k,
                "num_train_graphs": len(train_set),
                "num_test_graphs": len(test_set),
                "pos_weight": f"{pos_weight:.6f}",
                "test_binary_f1": f"{test_results['binary_f1']:.6f}",
                "test_macro_f1": f"{test_results['macro_f1']:.6f}",
                "zero_day_accuracy": f"{zero_day_results['accuracy']:.6f}",
                "train_loss": f"{history['train_losses'][-1]:.6f}",
                "seconds": f"{seconds:.1f}",
            })

            print(
                f"   k={k:4d}  train={len(train_set):5d}  pos_weight={pos_weight:.3f}"
                f"  macro_F1={test_results['macro_f1']:.4f}"
                f"  zero_day_acc={zero_day_results['accuracy']:.4f}"
                f"  ({seconds:.0f}s)"
            )

    print(f"\nResults written to {path}")

    return F1_results, Accuracy_for_zeroDay


### Subsection 4: Plots ###

def plot_GNN_results(
    GNN_type,
    K_values,
    F1_results,
    Accuracy_results,
    output_dir=None,
    ylim=(0.0, 1.02)
):
    """
    Draws the two learning curves of one architecture, side by side: macro F1 over
    the whole test set, and accuracy over the zero-day family alone. One line per
    family, sharing an x axis of k and a y axis of 0 to 1.

    Both panels use the same fixed y range so the curves can be compared directly
    and a flat curve looks flat. Pass ylim=None to auto-scale instead.

    Inputs:
    --- GNN_type: string
    --- K_values: list of ints, the x axis
    --- F1_results: dict mapping family -> list of (k, macro F1)
    --- Accuracy_results: dict mapping family -> list of (k, accuracy)
    --- output_dir: Path or string, defaults to results/figures
    --- ylim: (low, high) tuple or None
    Output:
    --- saved_paths: list of Path, the files written
    """

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
         "Common + held-out zero-day graphs"),
        (axes[1], Accuracy_results, "Accuracy on the zero-day family only",
         "held-out graphs of that family"),
    ]

    families = list(F1_results.keys())

    for axis, results, title, subtitle in panels:

        for index, family in enumerate(families):
            points = sorted(results.get(family, []))

            if not points:
                continue

            xs = [point[0] for point in points]
            ys = [point[1] for point in points]

            axis.plot(
                xs, ys,
                color=SERIES_COLORS[index % len(SERIES_COLORS)],
                marker=SERIES_MARKERS[index % len(SERIES_MARKERS)],
                markersize=5.5,
                markeredgecolor=surface,
                markeredgewidth=0.8,
                linewidth=2.0,
                label=family,
                zorder=3,
            )

        axis.set_title(title, color=ink, pad=10, loc="left")
        axis.set_xlabel(f"k  —  zero-day graphs added to training\n{subtitle}")
        axis.set_xticks(K_values)
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
        f"{GNN_type.upper()} — incremental out-of-distribution learning curves",
        x=0.012, ha="left", fontsize=13, color=ink,
    )

    # One legend for both panels: five series is above the four-series limit for
    # direct labels, so identity is carried by the legend plus the marker shapes.
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles, labels,
        loc="lower center", ncol=len(labels), frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    figure.tight_layout(rect=(0, 0.06, 1, 0.94))

    saved_paths = []

    for suffix in ("png", "pdf"):
        target = output_dir / f"main_experiment_{GNN_type.upper()}.{suffix}"
        figure.savefig(target, dpi=200, bbox_inches="tight")
        saved_paths.append(target)

    plt.close(figure)

    print("Figures written to:")
    for target in saved_paths:
        print(f"--- {target}")

    return saved_paths
