from collections import defaultdict
import os
import zipfile
from pathlib import Path
import pickle
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import yaml


def one_hot_encode(a):
    return np.eye(a.max() + 1)[a]


def end_with_newline(s):
    if s.endswith("\n"):
        return s
    else:
        return s + "\n"


def get_path_from_config(key):
    """
    Returns the path from the config file. If the path starts with `./`, it is relative to the main directory. Otherwise, it is absolute.
    """
    path = read_config()[key]
    if path[:2] == "./":
        return get_cwd() / path[2:]
    else:
        return Path(path)


def get_cwd():
    return Path(__file__).parents[1]


def read_config():
    return yaml.safe_load(open(get_cwd() / "config.yaml"))


def save_to(obj, path):
    with open(path, "wb") as f:
        return pickle.dump(obj, f)


def load_from(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_all_subjects(saved=True):
    """
    Returns all subjects in the dataset.

    Parameters
    ----------
    saved : bool
        Whether to use hardcoded subjects or to get them from `bids_path`.

    Returns
    -------
    subjects : list
        List of all subjects.
    """
    if saved:
        return [
            "sub-101",
            "sub-105",
            "sub-107",
            "sub-108",
            "sub-109",
            "sub-112",
            "sub-113",
            "sub-119",
            "sub-120",
            "sub-121",
            "sub-123",
            "sub-124",
            "sub-125",
            "sub-126",
            "sub-127",
            "sub-129",
            "sub-130",
            "sub-131",
            "sub-133",
            "sub-134",
            "sub-139",
            "sub-140",
            "sub-141",
            "sub-142",
            "sub-143",
            "sub-144",
            "sub-145",
            "sub-146",
            "sub-147",
            "sub-148",
            "sub-149",
        ]
    else:
        return sorted(
            [
                str(fname)[-7:]
                for fname in get_path_from_config("bids_path").glob("sub-*")
            ]
        )


def get_all_subject_filenames(saved=True):
    """
    Returns all subject filenames in the dataset.

    Parameters
    ----------
    saved : bool
        Whether to use hardcoded subjects or to get them from `data/OD`.

    Returns
    -------
    filenames : list
        List of all subject filenames.
    """
    return [subject + ".txt" for subject in get_all_subjects(saved)]


def get_all_channels():
    """
    Returns a list of all haemo channels.
    """
    return [
        "S1_D1 hbo",
        "S1_D1 hbr",
        "S2_D1 hbo",
        "S2_D1 hbr",
        "S3_D1 hbo",
        "S3_D1 hbr",
        "S1_D2 hbo",
        "S1_D2 hbr",
        "S3_D2 hbo",
        "S3_D2 hbr",
        "S4_D2 hbo",
        "S4_D2 hbr",
        "S2_D3 hbo",
        "S2_D3 hbr",
        "S3_D3 hbo",
        "S3_D3 hbr",
        "S5_D3 hbo",
        "S5_D3 hbr",
        "S3_D4 hbo",
        "S3_D4 hbr",
        "S4_D4 hbo",
        "S4_D4 hbr",
        "S5_D4 hbo",
        "S5_D4 hbr",
        "S6_D5 hbo",
        "S6_D5 hbr",
        "S7_D5 hbo",
        "S7_D5 hbr",
        "S8_D5 hbo",
        "S8_D5 hbr",
        "S6_D6 hbo",
        "S6_D6 hbr",
        "S8_D6 hbo",
        "S8_D6 hbr",
        "S9_D6 hbo",
        "S9_D6 hbr",
        "S7_D7 hbo",
        "S7_D7 hbr",
        "S8_D7 hbo",
        "S8_D7 hbr",
        "S10_D7 hbo",
        "S10_D7 hbr",
        "S8_D8 hbo",
        "S8_D8 hbr",
        "S9_D8 hbo",
        "S9_D8 hbr",
        "S10_D8 hbo",
        "S10_D8 hbr",
    ]


def get_lr_channels():
    return [
        "r_hbo",
        "r_hbr",
        "l_hbo",
        "l_hbr",
    ]


def get_lr_tb_channels():
    return [
        "l_t_hbo",
        "l_t_hbr",
        "l_b_hbo",
        "l_b_hbr",
        "r_t_hbo",
        "r_t_hbr",
        "r_b_hbo",
        "r_b_hbr",
    ]


def get_channels_in_region(region):
    sensors = {
        "left": ["S1_", "S2", "S3", "S4", "S5"],
        "right": ["S6", "S7", "S8", "S9", "S10"],
        "anterior": ["S7", "S10", "D7", "S4", "S1_", "D2"],
        "posterior": ["S9", "S6", "D6", "S5", "S2", "D3"],
    }
    return [c for c in get_all_channels() if any([s in c for s in sensors[region]])]


def get_channels_by_lat(level):
    levels = {
        0: ["S10", "S9", "D8"],
        1: ["S8", "D7", "D6"],
        2: ["S7", "S6", "D5"],
        3: ["S5", "S4", "D4"],
        4: ["S3", "D3", "D2"],
        5: ["S2", "S1_", "D1"],
    }
    return [c for c in get_all_channels() if any([s in c for s in levels[level]])]


def get_channels_by_pos(level):
    levels = {
        0: ["S10", "S7", "D7", "S4", "S1_", "D2"],
        1: ["D8", "S8", "D5", "D4", "S3", "D1"],
        2: ["S9", "S6", "D6", "S5", "S2", "D3"],
    }
    return [c for c in get_all_channels() if any([s in c for s in levels[level]])]


def unzip(zip_path, dir_path):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        return zip_ref.extractall(dir_path)


def reverse_dict(d):
    return {v: k for k, v in d.items()}


def get_event_name_mapping_for_task(task, include_events="empe"):
    if task == "4_class":
        event_name_mapping = {
            "LANV": 0,
            "HANV": 1,
            "LAPV": 2,
            "HAPV": 3,
        }
    elif task == "b_eb":
        event_name_mapping = {
            "LANV": 0,
            "LAPV": 0,
            "HANV": 1,
            "HAPV": 1,
        }
    elif task == "b_pn":
        event_name_mapping = {
            "LANV": 0,
            "HANV": 0,
            "LAPV": 1,
            "HAPV": 1,
        }
    elif task == "b_pepb":
        event_name_mapping = {
            "LAPV": 2,
            "HAPV": 3,
        }
    elif task == "b_pene":
        event_name_mapping = {
            "HANV": 1,
            "HAPV": 3,
        }
    elif task == "b_penb":
        event_name_mapping = {
            "LANV": 0,
            "HAPV": 3,
        }
    elif task == "b_pbne":
        event_name_mapping = {
            "HANV": 1,
            "LAPV": 2,
        }
    elif task == "b_pbnb":
        event_name_mapping = {
            "LANV": 0,
            "LAPV": 2,
        }
    elif task == "b_nenb":
        event_name_mapping = {
            "LANV": 0,
            "HANV": 1,
        }
    elif task == "r_v":
        event_name_mapping = {
            "LANV": 0,
            "HANV": 1,
            "LAPV": 2,
            "HAPV": 3,
        }
    elif task == "r_a":
        event_name_mapping = {
            "LANV": 0,
            "HANV": 1,
            "LAPV": 2,
            "HAPV": 3,
        }
    elif task == "r_c":
        event_name_mapping = {
            "LANV": 0,
            "HANV": 1,
            "LAPV": 2,
            "HAPV": 3,
        }
    if include_events == "afim":
        event_name_mapping = {f"afim_{k}": v for k, v in event_name_mapping.items()}
    return event_name_mapping


def get_p_labels(task):
    """
    returns an ordered list of p_{condition} labels for each condition
    """
    if task == "4_class":
        return ["p_nb", "p_ne", "p_pb", "p_pe"]
    elif task == "b_eb":
        return ["p_b", "p_e"]
    elif task == "b_pn":
        return ["p_n", "p_p"]
    elif task == "b_pepb":
        return ["p_pb", "p_pe"]
    elif task == "b_pene":
        return ["p_ne", "p_pe"]
    elif task == "b_penb":
        return ["p_nb", "p_pe"]
    elif task == "b_pbne":
        return ["p_ne", "p_pb"]
    elif task == "b_pbnb":
        return ["p_nb", "p_pb"]
    elif task == "b_nenb":
        return ["p_nb", "p_ne"]


def get_all_tasks():
    return [
        "4_class",
        "b_eb",
        "b_pn",
        "b_pepb",
        "b_pene",
        "b_penb",
        "b_pbne",
        "b_pbnb",
        "b_nenb",
        "r_v",
        "r_a",
        "r_c",
    ]


def get_task_id_to_name():
    """
    Returns a dictionary mapping task id to task name used in reports.
    """
    return {
        "4_class": "4 class",
        "b_pn": "Valence",
        "b_eb": "Arousal",
        "b_pene": "HA Valence",
        "b_pbnb": "LA Valence",
    }


def get_condition_abbreviation_map(include_events="empe"):
    abbr_map = {
        "LANV": "LANV",
        "HANV": "HANV",
        "LAPV": "LAPV",
        "HAPV": "HAPV",
    }
    if include_events == "afim":
        abbr_map = {f"afim_{k}": f"afim_{v}" for k, v in abbr_map.items()}
    return abbr_map


def get_model_from_str(model_str):
    if model_str == "slda":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("slda", LinearDiscriminantAnalysis(shrinkage="auto", solver="lsqr")),
            ]
        )
    else:
        print("Using default model: slda")
        return get_model_from_str("slda")


def get_target_names(task):
    event_name_mapping = get_event_name_mapping_for_task(task)
    if task == "b_pn" or task == "b_eb":
        return [
            label.split(" ")[0]
            for label in reverse_dict(reverse_dict(event_name_mapping)).keys()
        ]
    else:
        return [*event_name_mapping.keys()]


def get_scorer_with_kwargs(scorer, **kwargs):
    def scorer_with_kwargs(y_true, y_pred):
        return scorer(y_true, y_pred, **kwargs)

    return scorer_with_kwargs


def one_hot_encode(a, n_classes=None):
    if n_classes is None:
        n_classes = a.max() + 1
    b = np.zeros((a.shape[0], n_classes))
    b[np.arange(a.shape[0]), a] = 1
    return b


def map_to_ix(a):
    """E.g. [1, 2, 4, 2] -> [0, 1, 3, 1]"""
    res = np.zeros_like(a)
    for i, val in enumerate(np.unique(a)):
        res[a == val] = i
    return res


def make_close_to_best_bold(x, percent=0.98):
    return np.where(x >= percent * np.nanmax(x.to_numpy()), f"font-weight: bold", None)


def make_top_n_bold(x, n=3):
    nth_largest = np.sort(x.to_numpy())[-n]
    return np.where(x >= nth_largest, f"font-weight: bold", None)


class RenameUnpickler(pickle.Unpickler):
    """
    A custom unpickler that renames classes on the fly.
    """

    def find_class(self, module, name, rename_from, rename_to):
        if module.startswith(rename_from):
            module = module.replace(rename_from, rename_to)
        return super(RenameUnpickler, self).find_class(module, name)


def renamed_load(file_path):
    with open(file_path, "rb") as f:
        return RenameUnpickler(f).load()


def parse_afim_metadata(event_metadata):
    # scenario instructions and scenarios are stored in tags
    event_metadata[["scenario_instruction", "scenario"]] = event_metadata[
        "tags"
    ].str.split("___", expand=True)
    # shorten scenario instruction
    event_metadata["scenario_instruction"] = (
        event_metadata["scenario_instruction"]
        .str.replace(r".*shown.*", "shown", regex=True)
        .str.replace(r".*think.*", "think", regex=True)
    )
    return event_metadata


def get_tasks():
    return ["4_class", "b_pn", "b_eb", "b_pene", "b_pbnb"]


def get_include_events_to_name():
    return {"empe": "emotional perception", "afim": "affective imagery"}


def get_classification_methos_to_name():
    return {"com": "cross-participant", "ind": "subject-specific"}


def get_clf_pretty_abbr():
    return {
        "lr": "LR",
        "svc": "gSVM",
        "lda": "LDA",
        "mlp": "2-NN",
        "rf": "RF",
        "lsvc": "lSVM",
        "knn": "KNN",
    }


def order_channels(channels):
    return [ch for ch in get_all_channels() if ch in channels]


def get_channel_locations(epochs):
    """
    Returns a dictionary mapping channel name to channel location.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs object containing channel locations.


    Returns
    -------
    dict
        Dictionary mapping channel name to channel location.
    """
    return {ch["ch_name"]: ch["loc"][:2] for ch in epochs.info["chs"]}
