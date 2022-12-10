"""
Processing the raw OD data.
"""

from itertools import compress
import numpy as np
import mne
import warnings

from nemo.utils import read_config

config = read_config()

def interpolate_bads_nirs(inst, method="nearest", exclude=(), verbose=None):
    """
    Added method='average_nearest' to this mne function. It takes the average of the nearest instead of picking the first.
    """
    from scipy.spatial.distance import pdist, squareform
    from mne.preprocessing.nirs import _channel_frequencies, _check_channels_ordered

    # Returns pick of all nirs and ensures channels are correctly ordered
    freqs = np.unique(_channel_frequencies(inst.info))
    picks_nirs = _check_channels_ordered(inst.info, freqs)
    picks_nirs = sorted(
        picks_nirs
    )  # in new versions of mne, _check_channels_ordered returns channels in different order than data
    if len(picks_nirs) == 0:
        return

    nirs_ch_names = [inst.info["ch_names"][p] for p in picks_nirs]
    nirs_ch_names = [ch for ch in nirs_ch_names if ch not in exclude]
    bads_nirs = [ch for ch in inst.info["bads"] if ch in nirs_ch_names]
    if len(bads_nirs) == 0:
        return
    picks_bad = mne.io.pick.pick_channels(inst.info["ch_names"], bads_nirs, exclude=[])
    bads_mask = [p in picks_bad for p in picks_nirs]

    chs = [inst.info["chs"][i] for i in picks_nirs]
    locs3d = np.array([ch["loc"][:3] for ch in chs])

    mne.utils._check_option("fnirs_method", method, ["nearest", "average_nearest"])

    if method == "nearest":

        dist = pdist(locs3d)
        dist = squareform(dist)

        for bad in picks_bad:
            dists_to_bad = dist[bad]
            # Ignore distances to self
            dists_to_bad[dists_to_bad == 0] = np.inf
            # Ignore distances to other bad channels
            dists_to_bad[bads_mask] = np.inf
            # Find closest remaining channels for same frequency
            closest_idx = np.argmin(dists_to_bad) + (bad % 2)
            inst._data[bad] = inst._data[closest_idx]

        inst.info["bads"] = [ch for ch in inst.info["bads"] if ch in exclude]

    elif method == "average_nearest":
        """
        Takes mean of all nearest channels instead of just one
        """

        dist = pdist(locs3d)
        dist = squareform(dist)

        for bad in picks_bad:
            dists_to_bad = dist[bad]
            # Ignore distances to self
            dists_to_bad[dists_to_bad == 0] = np.inf
            # Ignore distances to other bad channels
            dists_to_bad[bads_mask] = np.inf
            # Find closest remaining channels
            all_closest_idxs = np.argwhere(
                np.isclose(dists_to_bad, np.min(dists_to_bad))
            )
            # Filter for same frequency as bad
            all_closest_idxs = all_closest_idxs[all_closest_idxs % 2 == bad % 2]
            inst._data[bad] = np.mean(inst._data[all_closest_idxs], axis=0)

        inst.info["bads"] = [ch for ch in inst.info["bads"] if ch in exclude]

    return inst


def process_raw(
    raw: mne.io.Raw,
    ch_interpolation=config["ch_interpolation"],
    sci_threshold=config["sci_threshold"],
    tddr=config["tddr"],
    l_freq=config["l_freq"],
    l_trans_bandwidth=config["l_trans_bandwidth"],
    h_freq=config["h_freq"],
    h_trans_bandwidth=config["h_trans_bandwidth"],
    bll_ppf=config["bll_ppf"],
    verbose=False,
) -> mne.io.Raw:
    """
    Applies preprocesing to a raw object.

    1. Detects bad channels with SCI and interpolates them. Use ``ch_interpolation=None`` to not interpolate.
    2. Applies TDDR to the raw object.
    3. Converts the optical density data to haemoglobin concentration using the modified Beer-Lambert law.
    4. Applies a band-pass (l_freq=0.01, h_freq=0.1) filter to the raw object
    
    Parameters
    ----------
    file_path : str
        Path to the file to read.
    data_type : str
        Type of data to read. Only 'OD' is currently supported.
    ch_interpolation : str
        Method to use to for channel interpolation.
    include_events : str or list, default "empe"
        Events to include. Codes are defined in data/triggers.txt. Only "empe" is currently supported.
    tddr : bool, default True
        Whether to apply Temporal Derivative Distribution Repair (TDDR).
    sci_threshold : float, default 0.5
        Threshold for the Scalp Coupling Index (SCI).
    l_freq : float, default 0.01
        Low cut-off frequency for the band-pass filter.
    h_freq : float, default 0.1
        High cut-off frequency for the band-pass filter.
    bll_ppf : int, default 6
        PPF for the modified Beer-Lambert law.
    """

    with np.errstate(
        invalid="ignore"
    ):  # some channels have all zeros, they will be eliminated
        sci = mne.preprocessing.nirs.scalp_coupling_index(raw)
    raw.info["bads"] = list(
        compress(raw.ch_names, (np.isnan(sci) | (sci < sci_threshold)))
    )
    if verbose:
        print(f'{len(raw.info["bads"])} channels marked as bad.')

    if ch_interpolation == "interpolate_average_nearest":
        interpolate_bads_nirs(raw, method="average_nearest")
    elif ch_interpolation == "interpolate_nearest":
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            raw.interpolate_bads()
    elif ch_interpolation == "drop":
        raw.drop_channels(raw.info["bads"])

    if tddr:
        raw = mne.preprocessing.nirs.temporal_derivative_distribution_repair(raw)

    # best practice is to use ppf=6, see https://github.com/mne-tools/mne-python/pull/9843
    raw = mne.preprocessing.nirs.beer_lambert_law(raw, ppf=bll_ppf)

    raw = raw.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        l_trans_bandwidth=l_trans_bandwidth,
        h_trans_bandwidth=h_trans_bandwidth,
        verbose=False,
    )
    return raw
