"""
Export epochs data from mne format to csv.
"""
#%%
import argparse
from nemo.epochs import get_epochs, get_epochs_dfs
from nemo.utils import get_cwd

#%%
parser = argparse.ArgumentParser()
parser.add_argument(
    "-e",
    "--include_events",
    nargs="+",
    default=["empe", "afim"],
    help="Event(s) to include (default: E, R)",
)
parser.add_argument(
    "-b",
    "--include_baseline",
    action="store_true",
    help="Include baseline (default: False)",
)
args = parser.parse_args()

for include_events in args.include_events:
    epochs_df, epochs_metadata = get_epochs_dfs(
        get_epochs(include_events=include_events)
    )

    if not args.include_baseline:
        epochs_df = epochs_df[epochs_df["time"] > 0]

    csv_data_dir = get_cwd() / "data" / f"{include_events}_csv"
    csv_data_dir.mkdir(parents=True, exist_ok=True)
    epochs_df.to_csv(
        csv_data_dir / "epochs.csv", sep=";", index=False
    )
    epochs_metadata.to_csv(
        csv_data_dir / "epochs_metadata.csv", sep=";", index=False
    )
