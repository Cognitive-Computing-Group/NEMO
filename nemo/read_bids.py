"""
Reads the BIDS data converts it to MNE objects.
"""
import pandas as pd
import mne
import logging
from mne_bids import BIDSPath, read_raw_bids

from nemo.utils import get_path_from_config, get_cwd, get_all_subjects
from nemo.process_raw import process_raw
from nemo.epochs import create_epochs_from_raw

logger = logging.getLogger(__name__)


def read_events_metadata(subject, include_events):
    """
    Reads a subject's raw optical density data from BIDS and returns it as a dataframe.
    """
    subject_id = subject[-3:]
    sub_events_path = (
        get_path_from_config("bids_path")
        / f"sub-{subject_id}"
        / "nirs"
        / f"sub-{subject_id}_task-{include_events}_events.tsv"
    )
    return pd.read_csv(sub_events_path, sep="\t")


def read_raw_od(subject, include_events):
    """
    Reads a subject's raw optical density data from BIDS and returns it as an MNE object.
    """
    subject_id = subject[-3:]
    bidspath = BIDSPath(
        subject=subject_id,
        task=include_events,
        root=get_path_from_config("bids_path"),
        datatype="nirs",
    )
    # mne_bids does not currently support reading fnirs_od, so we have to manually set the channel types and ignore warnings
    with mne.utils.use_log_level("ERROR"):
        raw_od_bids = read_raw_bids(bidspath).load_data()
        ch_map = {ch: "fnirs_od" for ch in raw_od_bids.ch_names}
        raw_od_bids.set_channel_types(ch_map)
    return raw_od_bids


def bids_to_mne(
    subjects=get_all_subjects(),
    include_events=["empe", "afim"],
    save_od=False,
    save_haemo=False,
    save_epochs=True,
):
    """
    Reads the BIDS data and creates and saves MNE objects in specified formats.

    Parameters
    ----------

    subjects : list of str, default: all subjects
        List of subject IDs to process.
    include_events : list of str, default: ["empe", "afim"]
        List of event types to include in epochs.
    save_od : bool, default: False
        Whether to save unprocessed OD data.
    save_haemo : bool, default: False
        Whether to save unprocessed haemoglobin data.
    save_epochs : bool, default: True
        Whether to save epochs.

    """
    if save_od:
        od_fif_path = get_cwd() / "processed_data" / "od"
        od_fif_path.mkdir(exist_ok=True)
    if save_haemo:
        haemo_path = get_cwd() / "processed_data" / "haemo"
        haemo_path.mkdir(exist_ok=True)
    if save_epochs:
        epochs_path = get_path_from_config("epochs_path")
        epochs_path.mkdir(exist_ok=True)

    for inc_events in include_events:
        for subject in subjects:
            logger.info(f"Processing subject={subject}, events={inc_events}")
            event_metadata = read_events_metadata(subject, include_events=inc_events)
            raw_od = read_raw_od(subject, include_events=inc_events)
            if save_od:
                raw_od.save(
                    od_fif_path / f"{subject}_task-{inc_events}_od_raw.fif",
                    overwrite=True,
                )
                logger.info(
                    f'Saved raw_od to {od_fif_path/f"{subject}_task-{inc_events}_od_raw.fif"}'
                )
                if not save_haemo and not save_epochs:
                    continue

            raw_haemo = process_raw(raw_od)
            if save_haemo:
                raw_haemo.save(
                    haemo_path / f"{subject}_task-{inc_events}_haemo_raw.fif",
                    overwrite=True,
                )
                logger.info(
                    f'Saved raw_haemo to {haemo_path/f"{subject}_task-{inc_events}_haemo_raw.fif"}'
                )
                if not save_epochs:
                    continue

            events, event_name_mapping = mne.events_from_annotations(raw_haemo)
            epochs = create_epochs_from_raw(
                raw_haemo,
                events=events,
                event_metadata=event_metadata,
                event_name_mapping=event_name_mapping,
            )
            if save_epochs:
                epochs.save(
                    epochs_path / f"{subject}_task-{inc_events}_epo.fif",
                    overwrite=True,
                    verbose="WARNING",
                )
                logger.info(
                    f'Saved epochs to {epochs_path/f"{subject}_task-{inc_events}_epo.fif"}'
                )
