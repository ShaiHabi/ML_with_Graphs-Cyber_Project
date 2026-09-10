# The main experiment: Out-of-Distribution and Learning Analysis.
#
# Training: Original + Distinct[zeroDay_type][train][:k]
# Validation: None (the held-out graphs of the zero-day family are in the test set
#             due to lack of data)
# Test: Common + Distinct[zeroDay_type][validation] + Distinct[zeroDay_type][test]
#
# All this file does is decide what those sets are. The sweep itself - every
# architecture, every k, the loss, the training and the scoring - lives in
# experiment_runner, which the baselines and the size-generalization control share.

# Project:
from experiment_runner import K_VALUES, experiment, plot_experiment_results
from preprocessed_data import get_common, get_distinct, get_original
from training import collect_graphs


### Global variables ###

# The five malware families of the "Distinct" dataset. It holds no benign graphs.
# The order fixes the order the families are swept in.
ZERODAYS_TYPES = ["clicker++trojan", "malware", "riskware", "spr", "spyware"]

def run_out_of_distribution_experiment(device=None, K_values=K_VALUES, resume=True):
    """
    Sweeps every zero-day family: for each one, trains on the whole Original dataset
    plus k of that family's graphs, and tests on the whole Common dataset plus the
    family's held-out graphs.

    Original and Common are flattened once. They are the same for every family and
    every k, and the graphs are shared rather than copied, so building them once is
    both the correct thing and what keeps the sweep in memory.

    Each family gets its own results CSV and its own pair of learning-curve panels.
    experiment() resumes from the CSV it wrote, so an interrupted sweep can simply be
    restarted and it only runs the (architecture, k) points still missing.

    Inputs:
    --- device: torch.device, defaults to cuda when available.
    --- K_values: iterable of ints.
    --- resume: bool.
    Outputs:
    --- F1_results: dict mapping family -> {architecture: [(k, macro F1), ...]}
    --- Accuracy_for_zeroDay: dict mapping family -> {architecture: [(k, accuracy), ...]}
    """
    experiment_name = "Out_of_Distribution"
    base_train_set = collect_graphs(get_original())   # all types, all splits
    base_test_set = collect_graphs(get_common())      # all types, all splits

    Distinct = get_distinct()

    F1_results = {}
    Accuracy_for_zeroDay = {}

    for zeroDay_type in ZERODAYS_TYPES:
        family_splits = Distinct[zeroDay_type]

        # The held-out part of the family. It is identical at every k, so the curve
        # measures the effect of k and nothing else.
        zeroDay_for_test = (
            list(family_splits["val"]) + list(family_splits["test"])
        )

        family_F1, family_accuracy = experiment(
            zeroDay_type,
            base_train_set,
            base_test_set,
            zeroDay_for_test,
            experiment_name=experiment_name,
            K_values=K_values,
            device=device,
            resume=resume
        )

        plot_experiment_results(
            zeroDay_type,
            K_values,
            family_F1,
            family_accuracy,
            experiment_name=experiment_name
        )

        F1_results[zeroDay_type] = family_F1
        Accuracy_for_zeroDay[zeroDay_type] = family_accuracy

    return F1_results, Accuracy_for_zeroDay


def run_dataset_analysis(base_test_set, device=None, K_values=[0], resume=True):
    experiment_name = "Dataset_Analysis"

    base_train_set = collect_graphs(get_original())
    F1_results = {}
    Accuracy_for_zeroDay = {}

    for zeroDay_type in [None]:
        # The held-out part of the family. It is identical at every k, so the curve
        # measures the effect of k and nothing else.
        zeroDay_for_test = []

        family_F1, family_accuracy = experiment(
            zeroDay_type,
            base_train_set,
            base_test_set,
            zeroDay_for_test,
            experiment_name=experiment_name,
            K_values=K_values,
            device=device,
            resume=resume
        )

        plot_experiment_results(
            zeroDay_type,
            K_values,
            family_F1,
            family_accuracy,
            experiment_name=experiment_name
        )

        F1_results[zeroDay_type] = family_F1
        Accuracy_for_zeroDay[zeroDay_type] = family_accuracy

    return F1_results, Accuracy_for_zeroDay


def run_per_family_analysis(device=None, K_values=K_VALUES, resume=True):
    """
    Checks the performance of the model on each family in the Distinct dataset.
    We use only benign graphs + k of the family for training, and the held-out graphs of that family for testing.
    Training Input:
    * base_train_set: 1400 benign graphs
    * k samples of the family being analyzed

    Testing Input:
    * base_test_set: 600 benign graphs
    * zeroDay_for_test: 300 held-out graphs of the family being analyzed
    """
    experiment_name = "Per_Family_Analysis"

    base_train_set = collect_graphs(get_original()["benign"]["train"]) + collect_graphs(get_common()["benign"]["train"])

    base_val_set = collect_graphs(get_original()["benign"]["val"]) + collect_graphs(get_common()["benign"]["val"])
    base_test_set = base_val_set + collect_graphs(get_original()["benign"]["test"]) + collect_graphs(get_common()["benign"]["test"])
    F1_results = {}
    Accuracy_for_zeroDay = {}

    for zeroDay_type in ZERODAYS_TYPES:
        zeroDay_for_test = collect_graphs(get_distinct()[zeroDay_type]["val"]) + collect_graphs(get_distinct()[zeroDay_type]["test"])

        family_F1, family_accuracy = experiment(
            zeroDay_type,
            base_train_set,
            base_test_set,
            zeroDay_for_test,
            experiment_name=experiment_name,
            K_values=K_values,
            device=device,
            resume=resume
        )

        plot_experiment_results(
            zeroDay_type,
            K_values,
            family_F1,
            family_accuracy,
            experiment_name=experiment_name
        )

        F1_results[zeroDay_type] = family_F1
        Accuracy_for_zeroDay[zeroDay_type] = family_accuracy

    return F1_results, Accuracy_for_zeroDay
