"""
Creates and saves a dataset. Requires parsed epochs, which can be created with scripts/load_bids.py.
"""
import argparse
import logging

from nemo.feature_extraction import create_datasets_from_epochs_df
from nemo.epochs import get_epochs, get_epochs_dfs
from nemo.utils import read_config

config = read_config()
logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()
parser.add_argument("-w", "--n_windows", type=int, default=config["n_windows"])
parser.add_argument("-f", "--features", nargs="+", default=config["features"])
parser.add_argument("-t", "--task", type=str, default=config["task"])
parser.add_argument(
    "-e", "--include_events", type=str, default=config["include_events"]
)
parser.add_argument("-c", "--ch_selection", type=str, default=config["ch_selection"])
parser.add_argument("--log_level", type=str, default="INFO")
args = parser.parse_args()

logging.basicConfig(level=args.log_level)

create_datasets_from_epochs_df(
    get_epochs_dfs(get_epochs(include_events=args.include_events))[0],
    n_windows=args.n_windows,
    features=args.features,
    task=args.task,
    include_events=args.include_events,
    ch_selection=args.ch_selection,
    save=True,
)
