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
    Checks how fast the model converges on an unseen malware family.
    We train on the whole Original dataset + k of the family, and
    test on the whole Common dataset + the held-out graphs of that family. Every family
    in the Distinct dataset is swept in turn, and each one is run with all four GNNs
    (GCN, GIN, GAT, GT) so the curves can be compared architecture by architecture.
    Training Input:
    * base_train_set: 5000 Original graphs (1000 benign, 4000 malware)
    * k samples of the family being analyzed

    Testing Input:
    * base_test_set: 5000 Common graphs (1000 benign, 4000 malware)
    * zeroDay_for_test: 300 held-out graphs of the family being analyzed
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


def run_dataset_analysis(
    base_test_set,
    analysis_name,
    device=None,
    K_values=(0,),
    resume=True
):
    """
    Checks how good the Original dataset already is on its own, without adding any
    zero-day samples. This is the control experiment for the other two: nothing is
    changed, so it gives the k = 0 reference point their curves are measured against,
    and it tells us whether adding the zero-day samples helps at all. The test set is
    chosen by the caller - main.py runs it once on Common and once on Distinct +
    Common's benign graphs.
    Training Input:
    * base_train_set: 5000 Original graphs (1000 benign, 4000 malware)
    * no zero-day samples, k is 0

    Testing Input:
    * base_test_set: passed in by the caller
    * zeroDay_for_test: empty, there is no family being analyzed
    """

    experiment_name = "Dataset_Analysis"

    K_values = tuple(K_values)

    if any(k > 0 for k in K_values):
        raise ValueError(
            "run_dataset_analysis has no zero-day family to draw samples from, so "
            f"every k has to be 0. Got {K_values}."
        )

    base_train_set = collect_graphs(get_original())

    # There is no family being analyzed here, so nothing is held out of one.
    zeroDay_for_test = []

    analysis_F1, analysis_accuracy = experiment(
        analysis_name,
        base_train_set,
        base_test_set,
        zeroDay_for_test,
        experiment_name=experiment_name,
        K_values=K_values,
        device=device,
        resume=resume
    )

    plot_experiment_results(
        analysis_name,
        K_values,
        analysis_F1,
        analysis_accuracy,
        experiment_name=experiment_name
    )

    # Keyed the way the other two drivers key their returns, so a caller can treat
    # all three alike.
    return {analysis_name: analysis_F1}, {analysis_name: analysis_accuracy}


def run_per_family_analysis(device=None, K_values=K_VALUES, resume=True):
    """
    Checks the performance of the model on each family in the Distinct dataset.
    We use only benign graphs + k of the family for training, and the held-out graphs of that family for testing.

    Every set below is built through collect_graphs' own malware_types and splits
    filters rather than by indexing into the dataset first. collect_graphs takes the
    whole {malware_type: {split: subset}} dict and calls .keys() on it, so handing it
    dataset["benign"]["train"] hands it a PyG subset, which has no .keys() at all.

    The grid keeps its explicit 0: that point trains on the 1,400 benign graphs alone,
    and is the reference the rest of the curve is read against. build_loss_function is
    what makes it runnable - a training set with no positives gets pos_weight 1 rather
    than a division by zero.

    Training Input:
    * base_train_set: 1400 benign graphs
    * k samples of the family being analyzed

    Testing Input:
    * base_test_set: 600 benign graphs
    * zeroDay_for_test: 300 held-out graphs of the family being analyzed
    """
    experiment_name = "Per_Family_Analysis"

    K_values = tuple(K_values)

    base_train_set = (
        collect_graphs(get_original(), malware_types=["benign"], splits=["train"])
        + collect_graphs(get_common(), malware_types=["benign"], splits=["train"])
    )

    base_test_set = (
        collect_graphs(get_original(), malware_types=["benign"], splits=["val", "test"])
        + collect_graphs(get_common(), malware_types=["benign"], splits=["val", "test"])
    )

    F1_results = {}
    Accuracy_for_zeroDay = {}

    for zeroDay_type in ZERODAYS_TYPES:
        zeroDay_for_test = collect_graphs(
            get_distinct(),
            malware_types=[zeroDay_type],
            splits=["val", "test"]
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
