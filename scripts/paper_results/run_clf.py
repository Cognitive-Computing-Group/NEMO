"""
Runs and saves classification results for further analysis.
"""
import argparse
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score

from nemo.classification import get_cv
from nemo.utils import get_cwd
from nemo.feature_extraction import load_dataset, create_experiment_id
from nemo.classification import get_clf_method, get_clfs

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--task", type=str, default="4_class")
parser.add_argument("-m", "--method", type=str, default="ind")
parser.add_argument("-p", "--permutation_seed", type=int, default=None)
parser.add_argument("-c", "--ch_selection", type=str, default="hbo")
parser.add_argument("-e", "--include_events", type=str, default="empe")
parser.add_argument("--log_level", type=str, default="INFO")
parser.add_argument("--save", action="store_true")
args = parser.parse_args()

logger = logging.getLogger(__name__)
logging.basicConfig(level=args.log_level)
permute_y = args.permutation_seed is not None
include_events = args.include_events
clf_method = get_clf_method(args.method)
scorer_str = "accuracy_score"
scorer = accuracy_score
cv = LeaveOneOut() if args.method == "com" else get_cv
task = args.task
X, y, _, = load_dataset(
    create_experiment_id(
        n_windows=3,
        features=["MV"],
        task=task,
        include_events=include_events,
        ch_selection=args.ch_selection,
    )
)
n_classes = len(np.unique(np.concatenate([*y.values()])))
clfs = get_clfs(method=args.method, n_classes=n_classes, include_events=include_events)
subjects = np.array([*X.keys()])

logger.info(
    f"Running {args.method} classification for {include_events} {task} task with {len(subjects)} subjects. X: {np.concatenate([*X.values()]).shape}"
)

all_scores = []
for model_name, model in clfs.items():
    model_scores = clf_method(
        X,
        y,
        model,
        cv,
        scorer,
        permute_y=permute_y,
        permutation_seed=args.permutation_seed,
    )
    for subject, score in model_scores.items():
        all_scores.append(
            [
                task,
                model_name,
                str(clf_method)[10:-16],
                subject,
                score,
                args.permutation_seed,
            ]
        )

    logger.info(
        f"{task:5} {model_name:4} {args.method} {scorer_str[:3]} perm_seed={args.permutation_seed} score: {np.mean([*model_scores.values()]):.3f}"
    )

if args.save:
    df = pd.DataFrame(
        all_scores,
        columns=["task", "model", "clf_method", "subject", "score", "permutation_seed"],
    )
    dest_dir = (
        get_cwd()
        / "processed_data"
        / "clf_scores"
        / include_events
        / args.method
        / task
    )
    dest_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        dest_dir
        / f'sdf{"_perm_" + str(args.permutation_seed) if permute_y else ""}.csv',
        index=False,
    )
