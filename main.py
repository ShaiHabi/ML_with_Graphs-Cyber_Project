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

from scripts.Out_of_Distribution import *
from scripts.training import RANDOM_STATE, set_random_seed, collect_graphs
from scripts.preprocessed_data import get_common, get_distinct, get_original

def main():
    """ Runs the project's experiments using the preprocessed datasets. """

    # Sets the random seed:
    set_random_seed(RANDOM_STATE)

    # Selects the available device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Random seed:", RANDOM_STATE)
    print("Device:", device)

    original = get_original()
    common = get_common()
    distinct = get_distinct()

    # The datasets are validated and loaded on first use, by preprocessed_data.
    # If a preprocessing output is missing it raises FileNotFoundError naming the
    # file and the command that produces it.
    print("Running Out-of-Distribution experiment...")
    run_out_of_distribution_experiment(device=device)

    print("Running dataset analysis on common graphs...")
    run_dataset_analysis(base_test_set=collect_graphs(common), device=device)

    print("Running dataset analysis on distinct + benign graphs...")
    run_dataset_analysis(base_test_set=collect_graphs(distinct) + collect_graphs(common["benign"]), device=device)

    print("Running per family analysis...")
    run_per_family_analysis(device=device)

if __name__ == "__main__":
    main()
