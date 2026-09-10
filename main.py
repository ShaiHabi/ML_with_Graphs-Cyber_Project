import os
import sys
import torch

# Global variables:
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_PATH = os.path.join(PROJECT_PATH, "scripts")

# The experiment modules import one another by plain module name, so the scripts
# directory has to be importable on its own. This must happen before they are imported.
if SCRIPTS_PATH not in sys.path:
    sys.path.insert(0, SCRIPTS_PATH)

from Out_of_Distribution import (
    run_dataset_analysis,
    run_out_of_distribution_experiment,
    run_per_family_analysis,
)
from preprocessed_data import get_common, get_distinct
from training import RANDOM_STATE, collect_graphs, set_random_seed

def main():
    """ Runs the project's experiments using the preprocessed datasets. """

    # Sets the random seed:
    set_random_seed(RANDOM_STATE)

    # Selects the available device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Random seed:", RANDOM_STATE)
    print("Device:", device)

    # The datasets are validated and loaded on first use, by preprocessed_data.
    # If a preprocessing output is missing it raises FileNotFoundError naming the
    # file and the command that produces it.
    common = get_common()
    distinct = get_distinct()

    print("Running Out-of-Distribution experiment...")
    run_out_of_distribution_experiment(device=device)

    # Each analysis is named, because the name is what its results are filed under.
    # Two runs sharing one name share one CSV, and the second then resumes the first
    # one's rows instead of training.
    print("Running dataset analysis on common graphs...")
    run_dataset_analysis(
        base_test_set=collect_graphs(common),
        analysis_name="Common",
        device=device
    )

    # collect_graphs takes a whole {malware_type: {split: subset}} dataset and picks
    # the types out itself. Passing common["benign"] passes it one type's splits, and
    # its "is this type wanted" test then compares graphs against split names, matches
    # nothing and quietly returns an empty list - so this test set used to hold no
    # benign graph at all.
    print("Running dataset analysis on distinct + benign graphs...")
    run_dataset_analysis(
        base_test_set=(
            collect_graphs(distinct)
            + collect_graphs(common, malware_types=["benign"])
        ),
        analysis_name="Distinct_and_benign",
        device=device
    )

    print("Running per family analysis...")
    run_per_family_analysis(device=device)

if __name__ == "__main__":
    main()
