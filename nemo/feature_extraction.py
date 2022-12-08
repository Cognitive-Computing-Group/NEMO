"""
Feature extraction. `create_datasets_from_epochs_df` can be used to create datasets (and save them with `save=True`).
"""
from collections import defaultdict
import logging
import numpy as np

from nemo.epochs import get_epochs_raw_dataset
from nemo.utils import (
    get_cwd,
    get_all_channels,
    load_from,
    save_to,
    read_config,
    order_channels,
)

config = read_config()
logger = logging.getLogger(__name__)


def extract_features_from_raw(X, features=["MV"], n_windows=3):
    Xf = defaultdict(list)
    for subject, d in X.items():
        sXf = []
        n_epochs, n_samples, n_channels = d.shape
        L = n_samples // n_windows
        for wi in range(n_windows):
            wd = d[:, wi * L : (wi + 1) * L, :]
            if "IAV" in features:
                sXf.append(np.sum(np.abs(wd), axis=1))
            if "MAV" in features:
                sXf.append(np.mean(np.abs(wd), axis=1))
            if "MV" in features:
                sXf.append(np.mean(wd, axis=1))
            if "PMN" in features:
                mu = np.mean(wd, axis=1)
                centered_wd = wd - mu[:, None, :]
                PMN = 0
                for si in range(wd.shape[1] - 1):
                    PMN += centered_wd[:, si, :] * centered_wd[:, si + 1, :] < 0
                sXf.append(PMN)
            if "PZN" in features:
                PZN = 0
                for si in range(wd.shape[1] - 1):
                    PZN += wd[:, si, :] * wd[:, si + 1, :] < 0
                sXf.append(PZN)
            if "STD" in features:
                sXf.append(np.std(wd, axis=1))
            if "polyfit_coef_1" in features:
                perm_wd = np.swapaxes(wd, 0, 1).reshape(wd.shape[1], -1)
                pf = np.polyfit(np.arange(perm_wd.shape[0]), perm_wd, 1)[0]
                sXf.append(pf.reshape(n_epochs, -1))
        sXf = (
            np.array(sXf).transpose(1, 2, 0).reshape(n_epochs, -1)
        )  # same order as old dataset
        Xf[subject] = sXf
    return Xf


def get_feature_names(features=["MV"], n_windows=3, ch_selection="hbo"):
    """
    Returns feature names in the same order as `extract_features_from_raw`.
    """
    channels = get_channels_by_selection(ch_selection)
    return np.array(
        [
            f"{ch} {f}_{wi:03d}"
            for ch in channels
            for wi in range(n_windows)
            for f in sorted(features)
        ]
    )


def get_channels_by_selection(ch_selection):
    if isinstance(ch_selection, str):
        channels = [ch for ch in get_all_channels() if ch_selection in ch]
    else:
        channels = order_channels(ch_selection)
    return channels


def create_experiment_id(features, n_windows, task, include_events, ch_selection):
    """
    Create experiment from parameters.
    """
    features_str = "_".join(features)
    ch_selection_str = str(ch_selection)
    return f"{include_events}-{task}-{features_str}-{n_windows}-{ch_selection_str}"


def create_datasets_from_epochs_df(
    epochs_df,
    features=config["features"],
    n_windows=config["n_windows"],
    task=config["task"],
    include_events=config["include_events"],
    ch_selection=config["ch_selection"],
    save=False,
    experiment_id=None,
    **kwargs,
):
    """
    Create datasets from epochs_df.

    Parameters
    ----------
    features : list, default=['MV', 'polyfit_coef_1']
        List of features to use. Possible features are: `MV`, `MAV`, `STD`, `polyfit_coef_1`, `PMN`, `PZN`, `IAV`.
    n_windows : int, default=1
        Number of windows to use.
    task : str, default='4_class'
        Task. Possible tasks are: `4_class`, `b_eb`, `b_pn`, `b_pepb`, `b_pene`, `b_penb`, `b_pbne`, `b_pbnb`, `b_nenb`, `r_v`, `r_a`, `r_c`.
    include_events : str, default="empe"
        Events to include. Possibilites are "empe": 'emotional perception', "afim": 'affective imagery'. Can include multiple types, e.g. 'ER'.
    discard_bad_epochs : bool, default=True
        Whether to discard bad epochs.
    bad_subject_threshold : float, default=0.
        Minimum portion of epochs that need to be good for subject to be included.
    ch_selection : str or list, default='hbo'
        Channel selection. Can be a list of channels or a substring of the channel name e.g. `hbo`.
    save : bool, default=False
        Whether to save the dataset to `processed_data/classification_datasets/{experiment_id}/`.
    experiment_id : str, default=None
        Experiment id. Used to run multiple experiments in parallel. If None, will be set to <include_events>_<task>_<features>_<n_windows>_<ch_selection>.

    Returns
    -------
    X : dict
        Dictionary subject -> numpy array.
    y : dict
        Dictionary subject -> numpy array.
    epoch_ids : dict
        Dictionary subject -> list of epoch_ids. Used for connecting metadata to rows in the numpy arrays.
    """
    channels = get_channels_by_selection(ch_selection)
    Xr, y, epoch_ids = get_epochs_raw_dataset(
        epochs_df, channels=channels, include_events=include_events, task=task, **kwargs
    )
    X = extract_features_from_raw(Xr, features=features, n_windows=n_windows)
    if save:
        experiment_id = (
            create_experiment_id(
                features, n_windows, task, include_events, ch_selection
            )
            if experiment_id is None
            else experiment_id
        )
        clf_dataset_dir = (
            get_cwd() / f"processed_data/classification_datasets/{experiment_id}"
        )
        clf_dataset_dir.mkdir(parents=True, exist_ok=True)
        save_to(X, clf_dataset_dir / "X.pkl")
        save_to(y, clf_dataset_dir / "y.pkl")
        save_to(epoch_ids, clf_dataset_dir / "epoch_ids.pkl")
        logger.info(f"Saved dataset to {clf_dataset_dir}")
    return X, y, epoch_ids


def load_dataset(experiment_id):
    clf_dataset_dir = (
        get_cwd() / f"processed_data/classification_datasets/{experiment_id}"
    )
    if not clf_dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset {experiment_id} not found. Create a dataset with `scripts/create_datasets.py`."
        )
    X = load_from(clf_dataset_dir / "X.pkl")
    y = load_from(clf_dataset_dir / "y.pkl")
    epoch_ids = load_from(clf_dataset_dir / "epoch_ids.pkl")
    return X, y, epoch_ids
