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

from Out_of_Distribution import run_out_of_distribution_experiment
from training import RANDOM_STATE, set_random_seed


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
    run_out_of_distribution_experiment(device=device)


if __name__ == "__main__":
    main()
