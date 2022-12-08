"""
Produces a score summary table from saved classification scores.
"""
#%%
import re
import pandas as pd
from nemo.utils import (
    get_classification_methos_to_name,
    get_clf_pretty_abbr,
    get_include_events_to_name,
    get_tasks,
    get_cwd,
    get_task_id_to_name,
    make_close_to_best_bold,
)
from nemo.classification import get_clf_method, get_clf_names

#%%
try:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--method", type=str, default="com")
    parser.add_argument("-e", "--include_events", type=str, default="empe")
    args = parser.parse_args()
    method, include_events = args.method, args.include_events
except:
    method, include_events = "ind", "empe"
#%%
n_decimals = 2
selected_tasks = get_tasks()
clf_method = get_clf_method(method)
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
#%%
score_table = (
    pd.concat(sdfs)
    .groupby(["task", "model"])
    .mean(numeric_only=True)
    .unstack()
    .score.loc[get_tasks()]
)
#%%
score_table.index = [get_task_id_to_name()[task] for task in score_table.index]
score_table = score_table[get_clf_names()]
score_table.columns = [get_clf_pretty_abbr()[clf] for clf in score_table.columns]

latex = (
    score_table.style.apply(make_close_to_best_bold, percent=0.975, axis=1)
    .format(f"{{:.{n_decimals}f}}")
    .to_latex(hrules=True)
)
latex = re.sub(r"\\font\-weightbold\s(\d\.\d\d)", r"\\textbf{\1}", latex)
#%%
table = rf"""\begin{{table}}[!ht]
\small
\centering
\setlength{{\tabcolsep}}{{0.43em}}
{latex}
\caption{{Accuracy for {get_include_events_to_name()[include_events]} {get_classification_methos_to_name()[method]} model classificication. Bold values are within 2.5\% of the best value on the same row ($\geq 0.975 * max$).}}
\label{{tab:{include_events}_{method}_clf}}
\end{{table}}
"""
print(table)
#%%
