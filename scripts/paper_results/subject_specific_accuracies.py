"""
Plots the subject-specific accuracies for the 4 class task.
"""
#%%
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import logging

from nemo.utils import get_cwd, get_include_events_to_name, get_clf_pretty_abbr

logger = logging.getLogger(__name__)
#%%
fig_dir = get_cwd() / "results" / "paper_results" / "sub_scores"
fig_dir.mkdir(parents=True, exist_ok=True)
for include_events in ["empe", "afim"]:
    for method in ["ind", "com"]:
        task = "4_class"
        sdf = pd.read_csv(
            get_cwd()
            / "processed_data"
            / "clf_scores"
            / include_events
            / method
            / task
            / "sdf.csv"
        )
        # best model for this setting
        clf = (
            sdf.groupby(["task", "model"])
            .mean(numeric_only=True)
            .unstack()
            .score.max()
            .idxmax()
        )
        sdf = sdf[
            (sdf["model"] == clf)
            & (sdf["clf_method"].str[:3] == method)
            & (sdf["permutation_seed"].isna())
            & (sdf["task"] == task)
        ]
        sdf = sdf.drop(
            columns=["model", "clf_method", "permutation_seed", "task"]
        ).sort_values("score", ascending=False)

        sns.set_theme(style="whitegrid")
        sns.set_context("paper")
        sns.set(font_scale=1.5)

        fig = plt.subplots(figsize=(10, 10), dpi=200)
        sns.barplot(x="score", y="subject", data=sdf, color="C0")
        plt.axvline(0.25, color="k", linestyle="--", label="Chance")
        plt.xlim(0, 0.6)
        plt.xlabel("Accuracy")
        plt.ylabel("Subject")
        plt.title(
            f'{"Individual" if method == "ind" else "Combined"} {get_include_events_to_name()[include_events]}'.capitalize()
            + f" {get_clf_pretty_abbr()[clf]}"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / f"{include_events}-{task}-{method}-best.png", dpi=200)
        plt.close()

        logger.info(f'Saved {fig_dir/f"{include_events}-{task}-{method}-best.png"}')
# %%
