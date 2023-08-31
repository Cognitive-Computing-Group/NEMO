"""
Produces a table with classification scores and significance level from the permutation test.
"""
#%%
import numpy as np
import pandas as pd
import re
from nemo.utils import (
    get_classification_methos_to_name,
    get_clf_pretty_abbr,
    get_include_events_to_name,
    get_tasks,
    get_cwd,
    get_task_id_to_name,
)
from nemo.classification import get_clf_method, get_clf_names
#%%
try:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--method", type=str, default="com")
    parser.add_argument("-e", "--include_events", type=str, default="empe")
    parser.add_argument("-n", "--n_permutations", type=int, default=1000)
    parser.add_argument("--multipletest_method", type=str, default="none")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    method, include_events = args.method, args.include_events
except:
    method, include_events, n_permutations, multipletest_method, verbose = (
        "com",
        "empe",
        1000,
        "none",
        False,
    )
#%%
n_decimals = 2
selected_tasks = get_tasks()
clf_method = get_clf_method(method)
models = get_clf_names()
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
# #%%


ps[ps < 0.01] = 2
ps[ps <= 1] = 3

p_symbols = {
    2: "*",
    3: ".",
}
ps = pd.DataFrame(ps, index=task_name_index, columns=models)
ps.columns = [get_clf_pretty_abbr()[clf] for clf in ps.columns]
ps = ps.replace(p_symbols)
#%%
score_table = (
    sdf
    .groupby(["task", "model"])
    .mean(numeric_only=True)
    .unstack()
    .score.loc[get_tasks()]
)
score_table = score_table[get_clf_names()]
score_table.columns = [get_clf_pretty_abbr()[clf] for clf in score_table.columns]
score_table.index = [get_task_id_to_name()[task] for task in score_table.index]
score_table = score_table.round(n_decimals)
# format add trailing zeros
score_table = score_table.applymap(lambda x: f"{x:3.2f}")
# add * to significant values in score table
score_table[ps == "*"] = score_table[ps == "*"] + r"$*$"
score_table
#%%
def make_close_to_best_bold_str(x, percent=0.98):
    """Custom styler to make strings with values close to the best value bold."""
    x_vals = re.findall(r"(\d+\.\d+)", x.to_string())
    x_vals = [float(x) for x in x_vals]
    return np.where(x_vals >= percent * np.nanmax(x_vals), f"font-weight: bold", None)

style = (
    score_table.style.apply(make_close_to_best_bold_str, percent=0.975, axis=1)
)

latex = style.to_latex(
    hrules=True,
    column_format="l" * (len(score_table.columns) + 1),
)

latex = re.sub(r"\\font\-weightbold\s*(\d\.\d+)", r"\\textbf{\1}", latex)

table = rf"""\begin{{table}}[!ht]
\small
\centering
\setlength{{\tabcolsep}}{{0.43em}}
{latex}
\caption{{
    Accuracy for {get_include_events_to_name()[include_events]} {get_classification_methos_to_name()[method]} classification. 
    \new{{Bold values are within 2.5\% of the highest value on the same row ($\geq0.975\,*$ highest value on the same row).
    Results with significant permutation test results are marked with $*$ ($p < 0.01$).}}
}}
\label{{tab:{include_events}_{method}_clf}}
\end{{table}}
"""
print(table)
#%%