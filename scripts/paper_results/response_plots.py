"""
Plots topoplots comparing the response of different conditions and joint plots for emotional perception and affective imagery.
"""
#%%
import warnings
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
import mne
import numpy as np

from nemo.epochs import get_epochs
from nemo.utils import (
    get_condition_abbreviation_map,
    get_cwd,
)

#%%
def get_gt_for_conditions(epochs_dict, conditions, picks):
    """Grand total, i.e. mean of subject-specific means"""
    gt_evoked = []
    for s_epochs in epochs_dict.values():
        s_gt = s_epochs.load_data()[conditions].pick(picks)._data.mean(axis=0)
        gt_evoked.append(s_gt)
    return np.array(gt_evoked).mean(axis=0)


#%%
fig_dir = get_cwd() / "results" / "paper_results" / "response_plots"
fig_dir.mkdir(parents=True, exist_ok=True)
picks = "hbo"
epochs_dict = dict(empe=get_epochs(include_events="empe"), afim=get_epochs(include_events="afim"))
epochs = {k: mne.concatenate_epochs(list(v.values())) for k, v in epochs_dict.items()}
#%% Topoplots
vmin = -0.1
vmax = -vmin
contours = np.linspace(vmin, vmax, 10 + 1)
topomap_args = {
    "extrapolate": "local",
    "time_format": "%d s",
    "vmin": vmin,
    "vmax": vmax,
    "contours": contours,
    "cmap": "RdBu_r",
    "colorbar": False,
    "cbar_fmt": "% 2.2f",
    "sensors": True,
}
ts_args = {
    "ylim": dict(hbo=[vmin, vmax]),
    "spatial_colors": True,
}
times = [1, 3, 5, 7, 9, 11]
picks = "hbo"
abbr_map = get_condition_abbreviation_map()

empe_conditions = [*epochs["empe"].event_id.keys()]
afim_conditions = [*epochs["afim"].event_id.keys()]

e_ha_conditions = [c for c in empe_conditions if "HA" in c]
e_la_conditions = [c for c in empe_conditions if "LA" in c]

e_pv_conditions = [c for c in empe_conditions if "PV" in c]
e_nv_conditions = [c for c in empe_conditions if "NV" in c]

r_ha_conditions = [c for c in afim_conditions if "HA" in c]
r_la_conditions = [c for c in afim_conditions if "LA" in c]

r_pv_conditions = [c for c in afim_conditions if "PV" in c]
r_nv_conditions = [c for c in afim_conditions if "NV" in c]

condition_groups = [
    ["empe", e_pv_conditions, e_nv_conditions, "PV - NV"],
    ["empe", e_ha_conditions, e_la_conditions, "HA - LA"],
    ["empe", ["LAPV"], ["LANV"], "LAPV - LANV"],
    ["empe", ["HAPV"], ["HANV"], "HAPV - HANV"],
    ["afim", r_pv_conditions, r_nv_conditions, "PV - NV"],
    ["afim", r_ha_conditions, r_la_conditions, "HA - LA"],
    ["afim", ["afim_LAPV"], ["afim_LANV"], "LAPV - LANV"],
    ["afim", ["afim_HAPV"], ["afim_HANV"], "HAPV - HANV"],
]
for gi, cgroup in enumerate(condition_groups):
    print(cgroup)
    include_events, conditions1, conditions2, title = cgroup
    plot_title = f"{title}"  # {picks}'

    avg1 = get_gt_for_conditions(epochs_dict[include_events], conditions1, picks=picks)
    avg2 = get_gt_for_conditions(epochs_dict[include_events], conditions2, picks=picks)
    diff = (
        epochs[include_events].average(picks=picks).copy()
    )  # create mne.evoked of correct dimensions
    diff.data = avg1 - avg2  # replace data with the grand average

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*namespace is deprecated")
        afig = diff.plot_topomap(
            times=times, **topomap_args, show=False, title=plot_title
        )
    afig.set_size_inches(5, 1.5)
    afig.texts[0].set_text(plot_title)
    afig.texts[0].set_rotation(90)
    afig.texts[0].set_fontsize(11)
    if len(plot_title) > 7:
        afig.texts[0].set_position((-0.16, 0.76))
    else:
        afig.texts[0].set_position((-0.16, 0.61))

    for i, ax in enumerate(afig.axes):
        if gi % 4 != 0:
            ax.set_title("")

        # enlarge topomaps
        delta = 0.08
        id = 0.05
        xinc = 0.03
        yinc = 0.04
        pos = ax.get_position(afig)
        x0, y0, x1, y1 = pos.x0, pos.y0, pos.x1, pos.y1
        ax.set_position(
            [
                x0 - delta + id * (i - 2) + xinc,
                y0 - delta + yinc,
                x1 - x0 + delta,
                y1 - y0 + delta,
            ]
        )

    afig.savefig(
        fig_dir / f"topo_{include_events}_{plot_title}.png",
        dpi=300,
        bbox_inches="tight",
        transparent=False,
    )
    plt.close(afig)

# take the colorbar
topomap_args["colorbar"] = True
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=r".*namespace is deprecated")
    afig = diff.plot_topomap(times=times, **topomap_args, show=False)
afig.set_size_inches(45, 25)
extent = afig.axes[-1].get_window_extent().transformed(afig.dpi_scale_trans.inverted())
extent.bounds = (
    extent.bounds[0] * 0.999,
    extent.bounds[1] * 0.9,
    extent.bounds[2] * 2.7,
    extent.bounds[3] * 1.13,
)
for ax in afig.axes[:-1]:
    ax.remove()
ax = afig.axes[-1]
fontsize = 25
ax.tick_params(axis="y", which="major", labelsize=fontsize)
ax.tick_params(axis="y", which="minor", labelsize=fontsize)
ax.title.set_fontsize(fontsize)
afig.savefig(fig_dir / f"colorbar.png", bbox_inches=extent, dpi=200)
#%% Joint plot
evoked_dict = {
    "Emotional perception": ["empe", empe_conditions],
    "Affective imagery": ["afim", afim_conditions],
}
topomap_args["contours"] = np.linspace(vmin, vmax, 4 + 1)

for name, (include_events, conditions) in evoked_dict.items():
    print(name, include_events, conditions)
    evoked = epochs[include_events][conditions].average(picks=picks)
    gt = get_gt_for_conditions(epochs_dict[include_events], conditions, picks=picks)
    evoked.data = gt
    afig = evoked.plot_joint(
        times=times,
        title=name,
        topomap_args={k: v for k, v in topomap_args.items() if k != "colorbar"},
        ts_args=ts_args,
        show=False,
    )

    afig.axes[0].texts[1].remove()
    afig.axes[0].set_xticks([-5, 0, *times, 12])
    afig.axes[0].axvline(0, color="k", linestyle="--", linewidth=1, alpha=0.5)

    afig.axes[-1].texts[0].set_fontsize(15)
    afig.axes[-1].texts[0].set_position((0.5, 0.35))

    delta = 0.09
    id = 0.03
    xinc = 0.0  # moves topos right
    yinc = 0.05  # moves topos up
    for i, ax in enumerate(afig.axes[1 : len(times) + 1]):
        # remove second title
        ax.set_title("")

        # enlarge topomaps
        pos = ax.get_position(afig)
        x0, y0, x1, y1 = pos.x0, pos.y0, pos.x1, pos.y1
        ax.set_position(
            [
                x0 - delta + id * (i - 2) + xinc,
                y0 - delta + yinc,
                x1 - x0 + delta,
                y1 - y0 + delta,
            ]
        )

        # adjust the pointer lines for new topomaps
        ld = afig.lines[i].get_xydata()
        ld[0, 0] = ld[0, 0] + id * (i - len(times) // 2) + xinc / 2 - 0.015
        ld[0, 1] = ld[0, 1] + yinc - 0.07
        afig.lines[i].set_data(ld[:, 0], ld[:, 1])

        # change cartoon head coords
        ip = InsetPosition(afig.axes[-2], [0.19, 0.15, 1.2, 1.05])
        afig.axes[-2].set_axes_locator(ip)

    event_type = "afim" if "afim" == conditions[0][0] else "empe"
    afig.savefig(
        fig_dir / f"joint_{event_type}_{name}.png",
        dpi=300,
        bbox_inches="tight",
        transparent=False,
    )
    plt.close(afig)
