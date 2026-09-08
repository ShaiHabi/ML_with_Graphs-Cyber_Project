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

EXPERIMENT_NAME = "Out_of_Distribution"


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

    print("\n=== Flattening Original and Common ===")
    base_train_set = collect_graphs(get_original())   # all types, all splits
    base_test_set = collect_graphs(get_common())      # all types, all splits
    print(f"Original={len(base_train_set)}, Common={len(base_test_set)}")

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
            experiment_name=EXPERIMENT_NAME,
            K_values=K_values,
            device=device,
            resume=resume
        )

        plot_experiment_results(
            zeroDay_type,
            K_values,
            family_F1,
            family_accuracy,
            experiment_name=EXPERIMENT_NAME
        )

        F1_results[zeroDay_type] = family_F1
        Accuracy_for_zeroDay[zeroDay_type] = family_accuracy

    return F1_results, Accuracy_for_zeroDay
