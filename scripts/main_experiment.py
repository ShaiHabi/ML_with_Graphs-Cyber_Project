# The main experiment: Out-of-Distribution and Learning Analysis.
#
# For every zero-day malware family in "Distinct" we train on the whole "Original"
# dataset plus the first k graphs of that family, and test on the whole "Common"
# dataset plus the family's held-out graphs. Increasing k answers the research
# question: how many labelled samples of a newly discovered family does the model
# need before it classifies the rest of that family reliably?


# General libraries
import csv
import os
import time
from pathlib import Path

# PyTorch:
import torch

# Project:
from GNNs_models import GNN_Model, evaluate_GNN_model
from training import (
    RANDOM_STATE,
    build_loss_function,
    collect_graphs,
    label_counts,
    make_loader,
    progress_bars,
    run_seed,
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
ZERODAYS_types = ["clicker++trojan", "malware", "riskware", "spr", "spyware"]

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

# Optimisation. epochs and batch_size are not in the plan's CONFIGURATION but a
# training loop cannot run without them.
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
    "test_accuracy",
    "test_precision",
    "test_recall",
    "test_roc_auc",
    "zero_day_accuracy",
    "train_loss",
    "seconds",
]


### Subsection 1: Model construction ###

def build_model(GNN_type, device, configuration=None):
    """
    Builds a GNN_Model from CONFIGURATION.

    CONFIGURATION also has to be validated here: GPSConv splits hidden_dim across
    the attention heads, so hidden_dim must divide evenly by heads or the failure
    surfaces much later as a shape error inside the attention layer.

    Inputs:
    --- GNN_type: string, one of "GCN", "GIN", "GAT", "GPS".
    --- device: torch.device
    --- configuration: dict, defaults to CONFIGURATION.
    Output:
    --- model: GNN_Model on the given device.
    """

    configuration = dict(CONFIGURATION if configuration is None else configuration)

    if GNN_type.upper() in {"GAT", "GPS"}:
        heads = configuration.get("heads")

        if not heads or configuration["hidden_dim"] % heads != 0:
            raise ValueError(
                f"{GNN_type} needs hidden_dim to be divisible by heads, got "
                f"hidden_dim={configuration['hidden_dim']} and heads={heads}."
            )

    model = GNN_Model(GNN_type, **configuration)

    return model.to(device)


def build_optimizer(model, training_configuration=None):
    """
    Builds the optimiser named in TRAINING_CONFIGURATION.

    Inputs:
    --- model: torch.nn.Module
    --- training_configuration: dict, defaults to TRAINING_CONFIGURATION.
    Output:
    --- optimizer: torch.optim.Optimizer
    """

    training_configuration = (
        TRAINING_CONFIGURATION if training_configuration is None
        else training_configuration
    )

    optimizer_name = str(training_configuration["optimizer"]).lower()
    learning_rate = training_configuration["learning_rate"]

    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate)

    if optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate)

    raise ValueError(f"Unsupported optimizer: {training_configuration['optimizer']}")


def batch_size_for(GNN_type, training_configuration=None):
    """
    Returns the batch size for one architecture, honouring BATCH_SIZE_BY_GNN_TYPE.

    Inputs:
    --- GNN_type: string
    --- training_configuration: dict, defaults to TRAINING_CONFIGURATION.
    Output:
    --- batch_size: int
    """

    training_configuration = (
        TRAINING_CONFIGURATION if training_configuration is None
        else training_configuration
    )

    return BATCH_SIZE_BY_GNN_TYPE.get(
        GNN_type.upper(), training_configuration["batch_size"]
    )


### Subsection 2: Results storage ###

def results_file_path(GNN_type, results_dir=None):
    """
    Returns the CSV path holding one architecture's results.

    Inputs:
    --- GNN_type: string
    --- results_dir: Path or string, defaults to results/main_experiment.
    Output:
    --- path: Path
    """

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
    k_values=None,
    zero_day_types=None,
    results_dir=None,
    resume=True,
    show_progress=False
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
    --- k_values: list of ints, defaults to K_VALUES.
    --- zero_day_types: list of family names, defaults to ZERODAYS_types.
    --- results_dir: Path or string for the CSV, defaults to results/main_experiment.
    --- resume: bool, skip (family, k) pairs already present in the CSV.
    --- show_progress: bool, show the per-batch tqdm bars.
    Outputs:
    --- F1_results: dict mapping family -> list of (k, binary F1 on the whole test set)
    --- Accuracy_for_zeroDay: dict mapping family -> list of (k, accuracy on the
        zero-day graphs only)
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    k_values = list(K_VALUES if k_values is None else k_values)
    zero_day_types = list(ZERODAYS_types if zero_day_types is None else zero_day_types)

    for name in ("Original", "Common", "Distinct"):
        if name not in datasets:
            raise KeyError(f"datasets is missing the {name!r} dataset.")

    path = results_file_path(GNN_type, results_dir)
    finished_runs = load_results(path) if resume else {}

    batch_size = batch_size_for(GNN_type)

    # Flattens the two datasets that do not change between runs. Building them once
    # matters: every k below reuses the same lists.
    print(f"\n=== Main experiment: {GNN_type.upper()} ===")
    print("Flattening Original and Common ...", end="", flush=True)
    original_graphs = collect_graphs(datasets["Original"])
    common_graphs = collect_graphs(datasets["Common"])
    print(f"Done! Original={len(original_graphs)}, Common={len(common_graphs)}")

    F1_results = {}
    Accuracy_for_zeroDay = {}

    for zero_day_type in zero_day_types:

        if zero_day_type not in datasets["Distinct"]:
            raise KeyError(
                f"{zero_day_type!r} is not a malware type of the Distinct dataset. "
                f"Available: {sorted(datasets['Distinct'].keys())}"
            )

        F1_results[zero_day_type] = []
        Accuracy_for_zeroDay[zero_day_type] = []

        family_splits = datasets["Distinct"][zero_day_type]

        # Step 3.1 - the held-out part of the family. It is identical at every k,
        # so the curve measures the effect of k and nothing else.
        zeroDay_for_test = (
            list(family_splits["val"]) + list(family_splits["test"])
        )
        zero_day_loader = make_loader(zeroDay_for_test, batch_size, shuffle=False)

        family_train_pool = family_splits["train"]

        print(
            f"\n-- {zero_day_type}: pool={len(family_train_pool)} graphs, "
            f"held out={len(zeroDay_for_test)} graphs"
        )

        for k in k_values:

            if k > len(family_train_pool):
                raise ValueError(
                    f"k={k} exceeds the {len(family_train_pool)} graphs in the "
                    f"train split of {zero_day_type!r}."
                )

            # Resume: reuse a run that already finished
            if (zero_day_type, k) in finished_runs:
                row = finished_runs[(zero_day_type, k)]
                F1_results[zero_day_type].append((k, float(row["test_binary_f1"])))
                Accuracy_for_zeroDay[zero_day_type].append(
                    (k, float(row["zero_day_accuracy"]))
                )
                print(f"   k={k:4d}  (already in {path.name}, skipped)")
                continue

            started_at = time.time()

            # Each point gets its own reproducible seed, so the curve is not shaped
            # by run order and a resumed sweep reproduces the earlier points.
            set_random_seed(run_seed(GNN_type, zero_day_type, k))

            # Step 3.2.1 - training set
            new_family_graphs = list(family_train_pool[:k]) if k > 0 else []
            train_set = original_graphs + new_family_graphs

            # Step 3.2.2 - test set
            test_set = common_graphs + zeroDay_for_test

            train_loader = make_loader(train_set, batch_size, shuffle=True)
            test_loader = make_loader(test_set, batch_size, shuffle=False)

            # Step 3.2.3 - loss, rebuilt for this k because pos_weight depends on it
            loss_function = build_loss_function(train_set, device)
            num_negative, num_positive = label_counts(train_set)
            pos_weight = num_negative / num_positive if num_positive else float("nan")

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
                show_progress=show_progress,
            )

            with progress_bars(show_progress):

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
            F1_results[zero_day_type].append((k, test_results["binary_f1"]))
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
                "test_accuracy": f"{test_results['accuracy']:.6f}",
                "test_precision": f"{test_results['precision']:.6f}",
                "test_recall": f"{test_results['recall']:.6f}",
                "test_roc_auc": f"{test_results['roc_auc']:.6f}",
                "zero_day_accuracy": f"{zero_day_results['accuracy']:.6f}",
                "train_loss": f"{history['train_losses'][-1]:.6f}",
                "seconds": f"{seconds:.1f}",
            })

            print(
                f"   k={k:4d}  train={len(train_set):5d}  pos_weight={pos_weight:.3f}"
                f"  F1={test_results['binary_f1']:.4f}"
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
    Draws the two learning curves of one architecture, side by side: F1 over the
    whole test set, and accuracy over the zero-day family alone. One line per
    family, sharing an x axis of k and a y axis of 0 to 1.

    Both panels use the same fixed y range so the curves can be compared directly
    and a flat curve looks flat. Pass ylim=None to auto-scale instead.

    Inputs:
    --- GNN_type: string
    --- K_values: list of ints, the x axis
    --- F1_results: dict mapping family -> list of (k, F1)
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
        (axes[0], F1_results, "F1 on the full test set",
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
