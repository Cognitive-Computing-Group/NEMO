"""
Runs permutation tests for all classification problems. Requires saved results for true and permuted classification scores.
"""
#%%
import numpy as np
import pandas as pd
import argparse

from nemo.classification import get_clf_method, get_clf_names
from nemo.utils import (
    get_cwd,
    get_task_id_to_name,
    get_classification_methos_to_name,
    get_clf_pretty_abbr,
    get_include_events_to_name,
    get_tasks,
)

#%%
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--method", type=str, default="com")
parser.add_argument("-e", "--include_events", type=str, default="empe")
parser.add_argument("-n", "--n_permutations", type=int, default=1000)
parser.add_argument("--multipletest_method", type=str, default="none")
parser.add_argument("-v", "--verbose", action="store_true")
args = parser.parse_args()
method, include_events, n_permutations, multipletest_method, verbose = (
    args.method,
    args.include_events,
    args.n_permutations,
    args.multipletest_method,
    args.verbose,
)

clf_method = get_clf_method(method)
selected_tasks = get_tasks()
models = get_clf_names()

if verbose:
    print(
        f"{include_events} {method}, n_permutations: {n_permutations} ({multipletest_method})"
    )

sdfs = [
    pd.read_csv(
        get_cwd()
        / "processed_data"
        / "clf_scores"
        / include_events
        / f"{clf_method.__name__[:3]}"
        / f"{task}"
        / "sdf.csv"
    )
    for task in selected_tasks
]
sdf = pd.concat(sdfs)

perm_sdfs = [
    pd.read_csv(
        get_cwd()
        / "processed_data"
        / "clf_scores"
        / include_events
        / f"{clf_method.__name__[:3]}"
        / f"{task}"
        / f"sdf_perm_{permutation_seed}.csv"
    )
    for task in selected_tasks
    for permutation_seed in range(1, n_permutations + 1)
]
perm_sdf = pd.concat(perm_sdfs)
#%%
Cs = np.zeros((len(selected_tasks), len(models)))
for ti, task in enumerate(selected_tasks):
    for mi, model in enumerate(models):
        tmdf = sdf.loc[(sdf.task == task) & (sdf.model == model)]
        perm_tmdf = perm_sdf.loc[(perm_sdf.task == task) & (perm_sdf.model == model)]
        true_score = tmdf.score.mean()
        perm_scores = (
            perm_tmdf.groupby("permutation_seed").mean(numeric_only=True).score.values
        )
        Cs[ti, mi] = np.sum(perm_scores >= true_score)
        if verbose:
            print(
                f"{task:7} {model:4} {true_score:3.2f} {np.mean(perm_scores):3.2f} {Cs[ti, mi]:3.0f}"
            )

csdf = pd.DataFrame(Cs, index=selected_tasks, columns=models)
task_name_index = [get_task_id_to_name()[task] for task in csdf.index]
csdf.index = task_name_index
if verbose:
    print(csdf.style.format("{:3.0f}").to_latex())

ps = (Cs + 1) / (n_permutations + 1)

if multipletest_method != "none":
    from statsmodels.stats.multitest import multipletests

    rejs, adj_ps, alpha_sidak, alpha_bonferroni = multipletests(
        ps.reshape(-1), alpha=0.05, method=multipletest_method, is_sorted=False
    )
    ps = adj_ps.reshape(ps.shape)

ps = pd.DataFrame(ps, index=selected_tasks, columns=models)
task_name_index = [get_task_id_to_name()[task] for task in ps.index]
ps.index = task_name_index
if verbose:
    print(ps.style.format("{:.3f}").to_latex())
ps[ps < 0.001] = 2
ps[ps < 0.01] = 3
ps[ps < 0.05] = 4
ps[ps <= 1] = 5
p_symbols = {
    2: "***",
    3: "**",
    4: "*",
    5: ".",
}
ps = pd.DataFrame(ps, index=task_name_index, columns=models)
ps.columns = [get_clf_pretty_abbr()[clf] for clf in ps.columns]
ps = ps.replace(p_symbols)

latex = ps.style.to_latex(hrules=True)

table = rf"""\begin{{table}}[!ht]
\small
\centering
\setlength{{\tabcolsep}}{{0.43em}}
{latex}
\caption{{Permutation test results for {get_include_events_to_name()[include_events]} {get_classification_methos_to_name()[method]} classification. The significance levels are marked with codes: $*$$**$ ($p < 0.001$), $**$ ($p < 0.01$), $*$ ($p < 0.05$), and $.$ ($p \geq 0.05$).}}
\label{{tab:{include_events}_{method}_permutation_test}}
\end{{table}}
"""
print(table)
#%%
