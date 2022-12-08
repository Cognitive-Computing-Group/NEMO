"""
Functions for working with epochs.
"""
import logging
from collections import defaultdict
import numpy as np
import pandas as pd
import mne
from tqdm import tqdm

from .utils import (
    get_all_subjects,
    get_event_name_mapping_for_task,
    get_path_from_config,
    read_config,
    get_all_channels,
)

logger = logging.getLogger(__name__)
config = read_config()


def create_epochs_from_raw(
    raw,
    events,
    event_name_mapping,
    event_metadata,
    tmin=-5,
    tmax=12,
    reject_criteria=dict(hbo=80e-6),
    verbose=False,
):
    """
    Creates epochs from raw data.

    1. Marks bad epochs based on maximum peak-to-peak signal amplitude (PTP)
    2. Saves `is_bad_epoch` and `bad_epoch_reason` to epochs.metadata.
    3. Removes epochs with missing data.

    Parameters
    ----------
    raw : mne.io.Raw
        Processed MNE raw object.
    events : array
        Events array.
    event_name_mapping : dict
        Mapping from event names to event codes.
    event_metadata : pd.DataFrame
        Event metadata.
    tmin : float, default -5
        Time before event to include in epoch.
    tmax : float, default 12
        Time after event to include in epoch.
    reject_criteria : dict, default dict(hbo=80e-6)
        Criteria for rejecting epochs. Keys are channel types and values are the maximum PTP.
    verbose : bool, default False
        Whether to print progress.

    Returns
    -------
    epochs : mne.Epochs
        The epochs object.
    """
    event_metadata["subject"] = raw.info["subject_info"]["his_id"]
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_name_mapping,
        metadata=event_metadata,
        tmin=tmin,
        tmax=tmax,
        verbose=False,
    )

    # only mark bad epochs since mne deletes bad epochs on save
    epochs.metadata["bad_epoch_reason"] = (
        epochs.copy().drop_bad(reject=reject_criteria, verbose=verbose).drop_log
    )
    epochs.metadata["is_bad_epoch"] = (
        epochs.metadata["bad_epoch_reason"].astype(str) != "()"
    )
    epochs.metadata["bad_channels"] = list(
        np.tile(epochs.info["bads"], (len(epochs.metadata), 1))
    )
    epochs.drop_bad(verbose=verbose)  # removes epochs with missing data

    return epochs


def get_epochs(
    subjects=get_all_subjects(),
    include_events=config["include_events"],
    disable_tqdm=True,
):
    """
    Gets MNE epoch objects from epoch_dir.

    Parameters
    ----------
    subjects : list
        List of subjects to read.
    include_events : list
        List of events to include.
    disable_tqdm : bool
        Disable tqdm progress bar.

    Returns
    -------
    epochs_dict: dict
        Dictionary subject -> epochs.
    """
    epochs_dict = {}
    for subject in tqdm(subjects, disable=disable_tqdm):
        try:
            epochs = mne.read_epochs(
                get_path_from_config("epochs_path")
                / f"{subject}_task-{include_events}_epo.fif",
                verbose="WARNING",
            )
            epochs_dict[subject] = epochs
        except FileNotFoundError:
            logger.warn(f"No epochs found for subject {subject}. Skipping.")
    if len(epochs_dict) == 0:
        raise FileNotFoundError("No epochs found. Have you run scripts/load_bids.py?")
    return epochs_dict


def combine_channels_lr(epochs):
    """
    Combine channels from left hemisphere and right hemisphere.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs object.

    Returns
    -------
    combined : mne.Epochs
        Epochs object with combined channels.
    """
    r_hbo_ix = np.arange(0, 24, 2)
    r_hbr_ix = np.arange(1, 24, 2)
    l_hbo_ix = np.arange(24, 48, 2)
    l_hbr_ix = np.arange(25, 48, 2)
    combined = mne.channels.combine_channels(
        epochs,
        groups=dict(
            r_hbo=r_hbo_ix,
            r_hbr=r_hbr_ix,
            l_hbo=l_hbo_ix,
            l_hbr=l_hbr_ix,
        ),
    ).apply_baseline()
    combined.event_id = epochs.event_id
    combined.metadata = epochs.metadata
    return combined


def combine_channels_lr_tb(epochs):
    """
    Combine channels from top-left hemisphere, top-right hemisphere, bottom-left hemisphere and bottom-right hemisphere.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs object.

    Returns
    -------
    combined : mne.Epochs
        Epochs object with combined channels.
    """
    l_t_hbo_ix = np.array([26, 28, 36, 38, 40, 42, 46])
    l_t_hbr_ix = l_t_hbo_ix + 1
    l_b_hbo_ix = np.array([24, 28, 30, 32, 34, 42, 44])
    l_b_hbr_ix = l_t_hbo_ix + 1
    r_t_hbo_ix = np.array([0, 4, 6, 8, 10, 18, 20])
    r_t_hbr_ix = r_t_hbo_ix + 1
    r_b_hbo_ix = np.array([2, 4, 12, 14, 16, 18, 22])
    r_b_hbr_ix = r_b_hbo_ix + 1
    combined = mne.channels.combine_channels(
        epochs,
        groups=dict(
            l_t_hbo=l_t_hbo_ix,
            l_t_hbr=l_t_hbr_ix,
            l_b_hbo=l_b_hbo_ix,
            l_b_hbr=l_b_hbr_ix,
            r_t_hbo=r_t_hbo_ix,
            r_t_hbr=r_t_hbr_ix,
            r_b_hbo=r_b_hbo_ix,
            r_b_hbr=r_b_hbr_ix,
        ),
    ).apply_baseline()
    combined.event_id = epochs.event_id
    combined.metadata = epochs.metadata
    return combined


def combine_epochs_channels(epochs_dict, combine_channels="lr_tb"):
    """
    Apply channel combination to epochs_dict.

    Parameters
    ----------
    epochs_dict: dict
        Dictionary subject -> epochs.
    combine_channels : str, default=`'lr_tb'`
        Channel combination to use. Possible values are `'lr'`, `'lr_tb'` and `None`.

    Returns
    -------
    epochs_dict: dict
        Combined epochs_dict. Subject -> epochs.
    """
    if combine_channels == "lr":
        epochs_dict = {
            subject: combine_channels_lr(epochs)
            for subject, epochs in epochs_dict.items()
        }
    elif combine_channels == "lr_tb":
        epochs_dict = {
            subject: combine_channels_lr_tb(epochs)
            for subject, epochs in epochs_dict.items()
        }
    return epochs_dict


def get_epochs_dfs(epochs_dict, disable_tqdm=True):
    """
    Combines epoch data and metadata from all subjects to dataframes `epochs_df` and `epochs_metadata_df`.

    Parameters
    ----------
    epochs_dict : dict
        Dictionary subject -> epochs.

    Returns
    -------
    epochs_df : pandas.DataFrame
        Dataframe containing all epoch data.
    epochs_metadata_df : pandas.DataFrame
        Dataframe containing all epoch metadata.
    """
    n_epochs = 0
    subject_edfs = []
    epoch_metadata_dfs = []
    for subject_epochs in tqdm(list(epochs_dict.values()), disable=disable_tqdm):
        subject_edf = subject_epochs.to_data_frame()
        subject_metadata = subject_epochs.metadata.copy()
        subject_edf["epoch"] += n_epochs  # new epoch ids
        subject_metadata["epoch"] += n_epochs
        subject_edf = pd.merge(subject_edf, subject_metadata, how="left", on="epoch")
        subject_edf = subject_edf[
            [
                "time",
                "subject",
                "epoch",
                "condition",
                "bad_channels",
                "is_bad_epoch",
                "bad_epoch_reason",
            ]
            + subject_epochs.ch_names
        ]
        subject_edfs.append(subject_edf)
        epoch_metadata_dfs.append(subject_metadata)
        n_epochs += len(subject_epochs)
    epochs_df = pd.concat(subject_edfs, ignore_index=True)
    epochs_metadata_df = pd.concat(epoch_metadata_dfs, ignore_index=True)
    return epochs_df, epochs_metadata_df


def get_epochs_raw_dataset(
    epochs_df,
    channels=None,
    remove_bad_epochs=True,
    bad_subject_threshold=0,
    include_events=config["include_events"],
    task="4_class",
):
    """
    Gets raw dataset from epochs_df.

    Parameters
    ----------
    epochs_df : pandas.DataFrame
        Dataframe containing all epoch data.
    channels : list, default None
        List of channels to include. If None, all channels are included.
    remove_bad_epochs : bool, default True
        Whether to remove bad epochs.
    bad_subject_threshold : float, default 0
        Threshold proportion of good epochs a subject needs to have to be included. Use 0 to not eliminate bad subjects.
    include_events : str or list, default "empe"
        Events to include.
    task : str, default '4_class'
        Task to use. `'4_class'` includes all information.

    Returns
    -------
    X : dict
        Dictionary subject -> numpy array of shape (n_epochs, n_samples, n_channels).
    y : dict
        Dictionary subject -> numpy array of shape (n_epochs,). The labels are: `0: 'LANV', 1: 'HANV', 2: 'LAPV', 3: 'HAPV'`.
    epoch_ids : dict
        Dictionary subject -> list of epoch_ids. Used for connecting metadata to rows in the numpy arrays.
    """
    event_mapping = get_event_name_mapping_for_task(task, include_events)
    task_epochs_df = epochs_df[epochs_df["condition"].isin(event_mapping.keys())]
    if channels is None:
        channels = get_all_channels()
    if remove_bad_epochs:
        included_epochs = get_good_epoch_ids(task_epochs_df, bad_subject_threshold)
    else:
        included_epochs = task_epochs_df["epoch"].unique()
    X = defaultdict(list)
    y = defaultdict(list)
    epoch_ids = defaultdict(list)
    for epoch_id in sorted(included_epochs):
        epoch_df = task_epochs_df[
            (task_epochs_df["epoch"] == epoch_id) & (task_epochs_df["time"] > 0)
        ]
        subject = epoch_df["subject"].iloc[0]
        X[subject].append(epoch_df[channels].values)
        y[subject].append(epoch_df["condition"].map(event_mapping).iloc[0])
        epoch_ids[subject].append(epoch_id)
    for subject in X.keys():
        X[subject] = np.array(X[subject])
        y[subject] = np.array(y[subject])
        epoch_ids[subject] = np.array(epoch_ids[subject])
    return X, y, epoch_ids


def get_good_epoch_ids(df, bad_subject_threshold=0.0):
    """
    Gets good epoch ids from `epochs_df` or `epochs_metadata_df`.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing epoch data.
    bad_subject_threshold : float, default 0
        Threshold proportion of good epochs a subject needs to have to be included. Use 0 to not eliminate bad subjects.
    """
    epochs_metadata_df = df[["epoch", "subject", "is_bad_epoch"]].drop_duplicates()
    good_epochs = epochs_metadata_df.loc[
        ~epochs_metadata_df["is_bad_epoch"], ["epoch", "subject"]
    ]
    if bad_subject_threshold == 0:
        return good_epochs["epoch"]
    # filter bad subjects
    n_subject_epochs = good_epochs["subject"].value_counts(sort=True)
    bad_subjects = n_subject_epochs[
        n_subject_epochs <= int(n_subject_epochs.max() * bad_subject_threshold)
    ].index.to_list()  # more than (1-bad_subject_threshold) of epochs are bad
    return epochs_metadata_df[
        ~(epochs_metadata_df["is_bad_epoch"])
        & ~(epochs_metadata_df["subject"].isin(bad_subjects))
    ]["epoch"].to_numpy()


def filter_bad_subjects(
    epochs_df, epochs_metadata_df, all_epochs, bad_subject_threshold=0
):
    """
    Filter out subjects with too many bad epochs.
    """
    ix = get_good_epoch_ids(
        epochs_metadata_df, bad_subject_threshold=bad_subject_threshold
    )
    filtered_epochs_metadata_df = epochs_metadata_df.copy().iloc[ix]
    filtered_epochs_df = epochs_df[
        epochs_df["epoch"].isin(filtered_epochs_metadata_df["epoch"])
    ]
    filtered_all_epochs = all_epochs.copy()[ix]
    return filtered_epochs_df, filtered_epochs_metadata_df, filtered_all_epochs


def mark_bad_subjects(
    epochs_df, epochs_metadata_df, all_epochs, bad_subject_threshold=0
):
    """
    Mark epochs from bad subjects as bad.
    """
    good_epochs = get_good_epoch_ids(
        epochs_metadata_df, bad_subject_threshold=bad_subject_threshold
    )
    marked_epochs_metadata_df = epochs_metadata_df.copy()
    marked_epochs_metadata_df.loc[
        ~marked_epochs_metadata_df["epoch"].isin(good_epochs), "is_bad_epoch"
    ] = True
    bad_epochs = marked_epochs_metadata_df.loc[
        marked_epochs_metadata_df["is_bad_epoch"], "epoch"
    ]
    marked_epochs_df = epochs_df.copy()
    marked_epochs_df["is_bad_epoch"] = marked_epochs_df["epoch"].isin(bad_epochs)
    return marked_epochs_df, marked_epochs_metadata_df, all_epochs


def get_label_map_from_metadata(epochs_metadata_df, label_field):
    """
    Returns a map from epoch ids to labels. E.g. `{0: 2.83, 1: 3.14, ...}`. Can be used as `y` in for e.g. regression.
    """
    return (
        epochs_metadata_df[["epoch", label_field]]
        .set_index("epoch")
        .astype(float)
        .to_dict()[label_field]
    )


def subtract_ch_mean(epochs):
    ch_mean = epochs.get_data().mean(axis=1)[:, None, :]
    new_epochs = epochs.copy()
    new_epochs._data -= ch_mean
    return new_epochs
