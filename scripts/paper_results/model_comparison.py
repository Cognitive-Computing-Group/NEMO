"""
Compares average performance of models.
"""
#%%
import numpy as np
import pandas as pd

from nemo.utils import (
    get_task_id_to_name,
    make_top_n_bold,
    get_cwd,
    get_clf_pretty_abbr,
    get_tasks,
)
from nemo.classification import get_clf_method, get_clf_names

#%%
score_tables = []
score_tables_ix = []
selected_tasks = get_tasks()
for include_events in ["empe", "afim"]:
    for method in ["ind", "com"]:
        clf_method = get_clf_method(method)
        sdfs = [
            pd.read_csv(
                get_cwd()
                / "processed_data"
                / "clf_scores"
                / include_events
                / f"{method}"
                / f"{task}"
                / "sdf.csv"
            )
            for task in selected_tasks
        ]
        score_table = (
            pd.concat(sdfs)
            .groupby(["task", "model"])
            .mean(numeric_only=True)
            .unstack()
            .score.loc[get_tasks()]
        )
        score_table.index = [get_task_id_to_name()[task] for task in score_table.index]
        score_table = score_table[get_clf_names()]
        score_table.columns = [
            get_clf_pretty_abbr()[clf] for clf in score_table.columns
        ]
        score_table.loc["4 class"] = score_table.loc["4 class"] * 2
        score_tables.append(score_table)
        score_tables_ix.append(f"{include_events}_{method}")
#%%
model_score_df = pd.DataFrame([t.mean() for t in score_tables], index=score_tables_ix)
model_score_df["mean"] = model_score_df.mean(axis=1)
model_score_df.loc["mean"] = model_score_df.mean(axis=0)
print(
    model_score_df.style.apply(make_top_n_bold, n=1, axis=1).format("{:.3f}").to_latex()
)
model_score_df.style.apply(make_top_n_bold, n=1, axis=1).format("{:.3f}")
