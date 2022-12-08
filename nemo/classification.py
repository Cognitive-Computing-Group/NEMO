#%%
"""
Functions for running classification experiments.
"""
from collections import defaultdict
import warnings
from joblib import Parallel, delayed
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.base import clone as clone_estimator
from sklearn.exceptions import ConvergenceWarning

from nemo.utils import (
    one_hot_encode,
    map_to_ix,
    get_cwd,
    load_from,
)


def get_cv(y, seed=1, **kwargs):
    _, label_counts = np.unique(y, return_counts=True)
    cv = RepeatedStratifiedKFold(
        n_splits=np.min(label_counts), n_repeats=1, random_state=seed, **kwargs
    )
    return cv


def get_clf_names():
    return [
        "lr",
        "svc",
        "lda",
        "mlp",
        "rf",
        "lsvc",
        "knn",
    ]


def get_clfs(include_events, method, n_classes=4):
    clfs = load_from(get_cwd() / "processed_data" / "models" / "paper_models.pkl")[
        include_events
    ][method]
    for name, model in clfs.items():
        if name == "lda":
            model.named_steps["clf"].priors = np.ones(n_classes) / n_classes
    return clfs


def get_base_clfs():
    clfs = dict(
        lr=LogisticRegression,
        svc=SVC,
        mlp=MLPClassifier,
        lda=LinearDiscriminantAnalysis,
        rf=RandomForestClassifier,
        lsvc=LinearSVC,
        knn=KNeighborsClassifier,
    )
    return clfs


def get_hard_voting(preds):
    return (preds == preds.max(axis=2)[:, :, None]).astype(int)


def permute_y_dict_comb(y_dict, seed=1):
    """Permute y in a subject --> label dict across subjects, while preserving the subject --> epoch structure"""
    rng = np.random.RandomState(seed=seed)
    y_long = []
    for subject, y_subj in y_dict.items():
        for y_subj_i in y_subj:
            y_long.append([subject, y_subj_i])
    y_long = np.array(y_long)
    y_long[:, 1] = rng.permutation(y_long[:, 1])
    # convert back to dict
    y_new = defaultdict(list)
    for subject, y in y_long:
        y_new[subject].append(y)
    for subject in y_new.keys():
        y_new[subject] = np.array(y_new[subject], dtype=int)
    return y_new


def get_subject_split(X, y, subjects, train_index, test_index):
    subjects_train, subjects_test = subjects[train_index], subjects[test_index]
    X_train_dict = {subject: X[subject] for subject in subjects_train}
    y_train_dict = {subject: y[subject] for subject in subjects_train}
    X_test_dict = {subject: X[subject] for subject in subjects_test}
    y_test_dict = {subject: y[subject] for subject in subjects_test}
    return X_train_dict, y_train_dict, X_test_dict, y_test_dict


def ensemble_clf(
    X, y, base_model, cv, scorer, permute_y=False, permutation_seed=None, verbose=0
):
    subjects = np.array(list(X.keys()))
    n_classes = len(np.unique(np.concatenate([*y.values()], axis=0)))
    scores = {}
    if permute_y:
        rng = np.random.RandomState(seed=permutation_seed)
        for subject in subjects:
            if permute_y:
                y[subject] = rng.permutation(y[subject])

    # train individual models
    models = {}
    for subject in subjects:
        models[subject] = clone_estimator(base_model).fit(X[subject], y[subject])

    scores = defaultdict(list)
    for train_index, test_index in cv.split(X):
        subjects_train, subjects_test = subjects[train_index], subjects[test_index]
        X_test_dict = {subject: X[subject] for subject in subjects_test}
        y_test_dict = {subject: y[subject] for subject in subjects_test}
        for test_subject in X_test_dict.keys():
            pred_dict = {}
            for train_subject in subjects_train:
                try:
                    pred_dict[train_subject] = models[train_subject].predict_proba(
                        X_test_dict[test_subject]
                    )
                except:
                    # if predict_proba is not available, convert to one-hot vectors
                    pred_dict[train_subject] = one_hot_encode(
                        map_to_ix(
                            models[train_subject].predict(X_test_dict[test_subject])
                        ),
                        n_classes=n_classes,
                    )
            preds = np.array([*pred_dict.values()])
            y_pred = preds.mean(axis=0).argmax(axis=1)
            y_true = map_to_ix(y_test_dict[test_subject])
            score = scorer(y_true, y_pred)
            scores[test_subject].append(score)

    for subj_scores in scores.values():
        if len(subj_scores) > 1:
            print("WARNING: Subject has more than one score")

    scores = {subject: np.array(scores[subject]).mean() for subject in scores.keys()}
    return scores


def combined_model_clf(
    X, y, base_model, cv, scorer, permute_y=False, permutation_seed=None, verbose=0
):
    if permute_y:
        y = permute_y_dict_comb(y, seed=permutation_seed)
    subjects = np.array(list(X.keys()))

    def calculate_split_score(train_index, test_index):
        X_train_dict, y_train_dict, X_test_dict, y_test_dict = get_subject_split(
            X, y, subjects, train_index, test_index
        )
        X_train = np.concatenate([*X_train_dict.values()], axis=0)
        y_train = np.concatenate([*y_train_dict.values()], axis=0)
        X_test = np.concatenate([*X_test_dict.values()], axis=0)
        y_test = np.concatenate([*y_test_dict.values()], axis=0)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            y_pred = clone_estimator(base_model).fit(X_train, y_train).predict(X_test)
        score = scorer(y_test, y_pred)
        return subjects[test_index], score

    scores = Parallel(n_jobs=-1)(
        delayed(calculate_split_score)(train_index, test_index)
        for train_index, test_index in cv.split(X)
    )

    for subjects, _ in scores:
        if len(subjects) > 1:
            print("WARNING: more than one subject in test set")
    scores = {subjects[0]: score for subjects, score in scores}
    if verbose >= 10:
        for subject, score in scores.items():
            print(f"{subject}: {score:.3f}")
    return scores


def individual_classification(
    X, y, base_model, cv, scorer, permute_y=False, permutation_seed=None, verbose=0
):
    subjects = np.array(list(X.keys()))
    scores = {}

    if permute_y:
        rng = np.random.RandomState(seed=permutation_seed)
        for subject in subjects:
            if permute_y:
                y[subject] = rng.permutation(y[subject])

    for subject in X.keys():
        if cv == get_cv:
            cvsub = get_cv(y[subject])
        else:
            cvsub = cv
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            score = cross_val_score(
                clone_estimator(base_model),
                X[subject],
                y[subject],
                cv=cvsub,
                scoring="accuracy",
                n_jobs=1,
            ).mean()
        scores[subject] = score
        if verbose >= 10:
            print(f"{subject}, {X[subject].shape}: {score:.3f}")
    return scores


def get_clf_method(method):
    if method == "ind":
        return individual_classification
    elif method == "com":
        return combined_model_clf
    elif method == "ens":
        return ensemble_clf
    else:
        raise ValueError("Invalid method")
