"""
Plots valence and arousal target values and highlights example images.
"""
#%%
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from nemo.epochs import get_epochs, get_epochs_dfs
from nemo.utils import get_condition_abbreviation_map, get_cwd

#%%
epochs_df, epochs_metadata_df = get_epochs_dfs(get_epochs())
epochs_metadata_df[["valence_mean", "arousal_mean"]] = epochs_metadata_df[
    ["valence_mean", "arousal_mean"]
].astype(float)
img_metadata_df = epochs_metadata_df.drop_duplicates("img_num").set_index("img_num")
img_metadata_df.index = img_metadata_df.index.astype(str)

# highlight example images
example_images = [
    "9001",
    "6260",
    "1604",
    "8190",
]

plt.style.use("dark_background")
plt.style.use("default")
plt.figure(figsize=(5, 5))

# correctly ordered for plotting
conditions = [
    "HANV",
    "LANV",
    "HAPV",
    "LAPV",
]
example_image_df = img_metadata_df.loc[example_images]

condition_abbreviation_map = get_condition_abbreviation_map()
for condition in conditions:
    cond_df = img_metadata_df[img_metadata_df["trial_type"] == condition]
    # move example images to the back
    cond_example_images = [img for img in example_images if img in cond_df.index]
    cond_example_df = cond_df.loc[cond_example_images]
    cond_df = cond_df.drop(cond_example_images)
    cond_df = pd.concat([cond_df, cond_example_df])
    sns.scatterplot(
        x="valence_mean",
        y="arousal_mean",
        data=cond_df,
        label=condition_abbreviation_map[condition],
        s=40,
        alpha=0.7,
    )


sns.scatterplot(
    x="valence_mean",
    y="arousal_mean",
    data=example_image_df,
    s=35,
    facecolor="none",
    edgecolor="black",
    linewidth=1,
)

h, l = plt.gca().get_legend_handles_labels()
legend_ix = [i for i, l in enumerate(l) if l in condition_abbreviation_map.values()]

plt.legend(
    np.array(h)[legend_ix],
    np.array(l)[legend_ix],
    ncol=2,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.00),
    framealpha=1,
)

val_mean = img_metadata_df["valence_mean"].mean()
aro_mean = img_metadata_df["arousal_mean"].mean()
val_max_diff = max(
    val_mean - img_metadata_df["valence_mean"].min(),
    img_metadata_df["valence_mean"].max() - val_mean,
)
aro_max_diff = max(
    aro_mean - img_metadata_df["arousal_mean"].min(),
    img_metadata_df["arousal_mean"].max() - aro_mean,
)

margin = 0.2
plt.xlim(val_mean - val_max_diff - margin, val_mean + val_max_diff + margin)
plt.ylim(aro_mean - aro_max_diff - margin, aro_mean + aro_max_diff + margin)

plt.axvline(val_mean, alpha=0.5, color="black")
plt.axhline(aro_mean, alpha=0.5, color="black")

plt.xlabel("Valence")
plt.ylabel("Arousal")
plt.tight_layout()
results_dir = get_cwd() / "results" / "paper_results" / "val_aro"
results_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(results_dir / "val_aro_plot.png", dpi=300)
# %%
