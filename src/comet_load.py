"""
Comet load.py
"""
import os
import math
import numpy as np
from typing import List, Tuple, Callable
import collections
import comet_ml
from comet_ml import API
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from src.constants import FIGURE_PATH, DATA_PATH, CONFIG_PATH, SYMBOL_DICT
from sithom.plot import plot_defaults, label_subplots, get_dim

comet_ml.config.save(api_key="57fHMWwvxUw6bvnjWLvRwSQFp")

comet_api = API()
print(comet_api.get())


def loop_through_project() -> None:
    workspace = "sdat2"  # /6dactive"
    project = "6dactive"

    for exp in comet_api.get(workspace, project):
        print("    processing project", project, "...")
        print("        processing experiment", exp.id, end="")
        print(".", end="")
        print(comet_api.get(workspace, project, exp.id))


METRICS = [
    ("inum", int),
    ("anum", int),
    ("mae", float),
    ("rmse", float),
    ("r2", float),
]


def loop_through_experiment(
    metrics: List[Tuple[str, Callable]],
    workspace: str = "sdat2",
    project: str = "6dactive",
) -> List[xr.Dataset]:
    # experiment = "f5e7f5b6d8c34f9b9c3d7e5a6d2f2c3"
    # experiment = "261f1786b8ab496e90170d593deba88f"
    ds_list = []
    for exp in comet_api.get(workspace, project):
        # exp = comet_api.get(workspace, project, experiment)
        print(dir(exp))
        metric_d = {}
        for metric, typ in metrics:
            metrics_l = exp.get_metrics(metric)
            metric_l = [typ(metrics_l[i]["metricValue"]) for i in range(len(metrics_l))]
            metric_d[metric] = (["point"], metric_l)
            print(exp.id, metric, "len(metrics)", len(metrics_l))
            # print("metrics", metrics)
            # for i in range(len(metrics)):
            #    print("metrics[" + str(i) + "]", metrics[i]["metricValue"])
        # print("metrics[0]", metrics[0]["metricValue"])
        # print(metric_d)
        ds = xr.Dataset(data_vars=metric_d)
        ds_list.append(ds)

    return ds_list


def plot_inum_metrics(ds_list: List[xr.Dataset]) -> None:
    plot_defaults()
    os.makedirs(os.path.join(FIGURE_PATH, "6dactive"), exist_ok=True)
    fig, axs = plt.subplots(3, 1, sharex=True)
    axs[0].set_ylabel("r$^{2}$ [-]")
    axs[1].set_ylabel("MAE [m]")
    axs[2].set_ylabel("RMSE [m]")
    axs[2].set_xlabel("Number of Latin Hypercube Samples")
    min_i = 1
    max_i = 1
    for ds in ds_list:
        ds = ds.where(ds.anum == 0, drop=True)
        axs[0].plot(ds.inum.values, ds.r2.values)
        # axs[0].plot(ds.r2)
        axs[1].plot(ds.inum.values, ds.mae)
        axs[2].plot(ds.inum.values, ds.rmse)
        try:
            tmp_max = ds.inum.max().values
        except:
            tmp_max = 1
        max_i = max(max_i, tmp_max)
    label_subplots(axs, y_pos=0.7, x_pos=0.05, fontsize=12)
    plt.xlim(min_i, max_i)

    plt.savefig(os.path.join(FIGURE_PATH, "6dactive", "inum_metrics.png"))
    plt.clf()

    fig, axs = plt.subplots(3, 1, sharex=True)
    axs[0].set_ylabel("(1 - r$^{2}$) [-]")
    axs[1].set_ylabel("MAE [m]")
    axs[2].set_ylabel("RMSE [m]")
    axs[2].set_xlabel("Number of Latin Hypercube Samples [-]")
    min_i = 1
    max_i = 1
    for ds in ds_list:
        ds = ds.where(ds.anum == 0, drop=True)
        axs[0].loglog(ds.inum.values, 1 - ds.r2.values)
        # axs[0].plot(ds.r2)
        axs[1].loglog(ds.inum.values, ds.mae)
        axs[2].loglog(ds.inum.values, ds.rmse)
        try:
            tmp_max = ds.inum.max().values
        except:
            tmp_max = 1
        max_i = max(max_i, tmp_max)

    plt.xlim(min_i, max_i)
    label_subplots(axs, y_pos=0.7, x_pos=0.05, fontsize=12)

    plt.savefig(os.path.join(FIGURE_PATH, "6dactive", "inum_metrics_log.png"))
    plt.clf()


def plot_final_metrics(ds_list: List[xr.Dataset]) -> None:
    plot_defaults()
    os.makedirs(os.path.join(FIGURE_PATH, "6dactive"), exist_ok=True)
    fig, axs = plt.subplots(1, 1, sharex=True)  # aspect=1)
    # axs[0].set_ylabel("r$^{2}$ [-]")
    # axs[1].set_ylabel("MAE [m]")
    # axs[2].set_ylabel("RMSE [m]")
    # axs[2].set_xlabel("Number of Latin Hypercube Samples")
    # min_i = 1
    # max_i = 1
    inum_l, anum_l, r2_l, mae_l, rmse_l = [], [], [], [], []
    for ds in ds_list:
        try:
            inum_l.append(ds.inum.values[-1])
            anum_l.append(ds.anum.values[-1])
            r2_l.append(ds.r2.values[-1])
            mae_l.append(ds.mae.values[-1])
            rmse_l.append(ds.rmse.values[-1])
        except:
            pass
        # axs[0].plot(ds.r2)
        # axs[1].plot(ds.inum.values[-1], ds.mae)
        # axs[2].plot(ds.inum.values, ds.rmse)
        # max_i = max(max_i, ds.inum.max().values)
    print(len(inum_l), len(anum_l), len(r2_l), len(mae_l), len(rmse_l))
    im = plt.scatter(inum_l, anum_l, c=r2_l, cmap="viridis", vmin=0, vmax=1)
    for j in [30, 60, 90, 120]:
        x = np.linspace(0, j, num=100)
        y = -x + j
        plt.plot(x, y, "k--")
    plt.xlim(0, 120)
    plt.ylim(0, 120)
    cbar = plt.colorbar(im, label="r$^{2}$ [-]")
    plt.xlabel("Number of Intial Latin Hypercube Samples [-]")
    plt.ylabel("Number of Actively Chosen Points [-]")
    # label_subplots(axs, y_pos=0.7, x_pos=0.05, fontsize=12)
    # plt.xlim(min_i, max_i)
    plt.savefig(os.path.join(FIGURE_PATH, "6dactive", "inum_anum_r2.png"))
    plt.clf()
    im = plt.scatter(inum_l, anum_l, c=rmse_l, cmap="viridis", vmin=0, vmax=1)
    for j in [30, 60, 90, 120]:
        x = np.linspace(0, j, num=100)
        y = -x + j
        plt.plot(x, y, "k--")
    plt.xlim(0, 120)
    plt.ylim(0, 120)
    cbar = plt.colorbar(im, label="RMSE [m]")
    plt.xlabel("Number of Intial Latin Hypercube Samples [-]")
    plt.ylabel("Number of Actively Chosen Points [-]")
    plt.savefig(os.path.join(FIGURE_PATH, "6dactive", "inum_anum_rmse.png"))
    plt.clf()
    im = plt.scatter(inum_l, anum_l, c=mae_l, cmap="viridis", vmin=0, vmax=1)
    for j in [30, 60, 90, 120]:
        x = np.linspace(0, j, num=100)
        y = -x + j
        plt.plot(x, y, "k--")
    plt.xlim(0, 120)
    plt.ylim(0, 120)
    cbar = plt.colorbar(im, label="MAE [m]")
    plt.xlabel("Number of Intial Latin Hypercube Samples [-]")
    plt.ylabel("Number of Actively Chosen Points [-]")
    plt.savefig(os.path.join(FIGURE_PATH, "6dactive", "inum_anum_mae.png"))
    plt.clf()

    # let's do a line plot instead.
    tnum = np.array(inum_l) + np.array(anum_l)
    frac = np.array(anum_l) / tnum
    r2 = np.array(r2_l)
    rmse = np.array(rmse_l)
    mae = np.array(mae_l)
    plt.plot(tnum, frac)
    plt.xlabel("Total Number of Samples [-]")
    plt.ylabel("Fraction of Actively Chosen Points [-]")
    # plt.show()
    plt.clf()
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=get_dim(ratio=1.5))
    axs[0].set_ylabel("r$^{2}$ [-]")
    axs[1].set_ylabel("MAE [m]")
    axs[2].set_ylabel("RMSE [m]")
    axs[2].set_xlabel("Number of Latin Hypercube Samples [-]")
    for j in [30, 60, 120]:
        frac_t = frac[tnum == j]
        r2_t = r2[tnum == j]
        idx = np.argsort(frac_t)
        frac_t = frac_t[idx]
        r2_t = r2_t[idx]
        rmse_t = rmse[tnum == j][idx]
        mae_t = mae[tnum == j][idx]
        axs[0].plot(frac_t, r2_t, label=f"{j}")
        axs[1].plot(frac_t, mae_t)
        axs[2].plot(frac_t, rmse_t)
    label_subplots(axs)
    plt.xlabel("Fraction of Actively Chosen Points [-]")
    axs[0].legend(title="Total Number of Samples [-]")
    axs[0].set_xlim(0, 1)
    plt.savefig(os.path.join(FIGURE_PATH, "6dactive", "frac_r2.png"))


def get_optima():
    pass


def rose_plot():
    # https://www.python-graph-gallery.com/391-radar-chart-with-several-individuals
    # Libraries
    import pandas as pd

    # Set data
    df = pd.DataFrame(
        {
            "group": ["A", "B", "C", "D"],
            "var1": [38, 1.5, 30, 4],
            "var2": [29, 10, 9, 34],
            "var3": [8, 39, 23, 24],
            "var4": [7, 31, 33, 14],
            "var5": [28, 15, 32, 14],
        }
    )

    # ------- PART 1: Create background

    # number of variable
    categories = list(df)[1:]
    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)

    # If you want the first axis to be on top:
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([10, 20, 30], ["10", "20", "30"], color="grey", size=7)
    plt.ylim(0, 40)

    # ------- PART 2: Add plots

    # Plot each individual = each line of the data
    # I don't make a loop, because plotting more than 3 groups makes the chart unreadable

    # Ind1
    values = df.loc[0].drop("group").values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle="solid", label="group A")
    ax.fill(angles, values, "b", alpha=0.1)

    # Ind2
    values = df.loc[1].drop("group").values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle="solid", label="group B")
    ax.fill(angles, values, "r", alpha=0.1)

    # Add legend
    plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))

    # Show the graph
    plt.show()


def active_learning_reliability_plot() -> None:
    # normalize the data
    # plot settings on rose plot
    # plot convergence.
    m = [
        # ("inum", int),
        # ("anum", int),
        ("angle", float),
        ("speed", float),
        ("point_east", float),
        ("rmax", float),
        ("pc", float),
        ("xn", float),
        ("max", float),
        ("step", int),
    ]
    ds_list = loop_through_experiment(
        m,
        workspace="sdat2",
        project="find-max-naive",
    )
    print(ds_list)

    ds_list = loop_through_experiment(
        [("INIT_SAMPLES", int), ("ACTIVE_SAMPLES", int)],
        workspace="sdat2",
        project="find-max-naive",
    )
    print(ds_list)
    ends_ds_list = []
    for i in range(len(ds_list)):
        # print(ds_list[0].isel(point=-1))
        ends_ds_list.append(
            ds_list[i]
            .isel(point=-1)
            .expand_dims(dim={"experiment": 1}, axis=-1)
            .assign_coords(experiment=("experiment", [i]))
        )
    print("ends_ds_list", ends_ds_list)
    ends_ds = xr.merge(ends_ds_list)
    print("ends_ds", ends_ds)
    ends_ds.to_netcdf(os.path.join(DATA_PATH, "ends_nm_ds.nc"))


def ansley_plot() -> None:
    LINES_MIX = ["b-", "r-", "g-", "m-", "c-", "k-"]
    LINES_ACTIVE = ["y--", "k--", "m--", "c--", "y..", "k..", "m..", "c.."]
    plot_defaults()
    api_out = get_evolution("find-max-naive")
    fig, axs = plt.subplots(1, 2)  # #figsize=get_dim(ratio=2))
    update_projection(axs, axs.flat[1], "polar")
    xs = [x + 1 for x in range(50)]
    for i, exp in enumerate(api_out):
        axs[0].plot(
            xs, api_out[exp]["metrics"][0]["values"], LINES_MIX[i], label=f"a{i+1}"
        )
    api_out = get_evolution("find-max-full-active")
    for i, exp in enumerate(api_out):
        axs[0].plot(
            xs, api_out[exp]["metrics"][0]["values"], LINES_ACTIVE[i], label=f"b{i+1}"
        )
    # axs[0].legend()
    axs[0].set_xlim(1, 50)
    axs[0].set_xlabel("Number of Samples")
    axs[0].set_ylabel("Maximum Surge Height [m]")
    slim_d = normalize_max_d(get_max("find-max-naive"))
    plot_slim_d(slim_d, axs.flat[1], LINES_MIX)
    slim_d = normalize_max_d(get_max("find-max-full-active"))
    plot_slim_d(slim_d, axs.flat[1], LINES_ACTIVE)
    label_subplots(axs)
    plt.savefig(os.path.join(FIGURE_PATH, "max_surge_ansley.png"))
    plt.clf()


def get_max(experiment="find-max-2") -> collections.defaultdict:
    api_out = comet_api.get_metrics_for_chart(
        [exp.id for exp in comet_api.get("sdat2", experiment)],
        ["angle", "speed", "point_east", "rmax", "pc", "xn", "max", "step"],
        ["init_samples", "active_samples", "seed"],
    )
    out_d = collections.defaultdict(list)
    for i, exp in enumerate(api_out):
        for metric in api_out[exp]["metrics"]:
            out_d[metric["metricName"]].append(metric["values"][-1])
        out_d["id"].append(exp)
    return out_d


def get_evolution(experiment="find-max-2"):
    return comet_api.get_metrics_for_chart(
        [exp.id for exp in comet_api.get("sdat2", experiment)],
        # ["angle", "speed", "point_east", "rmax", "pc", "xn", "max", "step"],
        ["max"],
        ["init_samples", "active_samples", "seed"],
    )


def update_projection(ax, axi, projection="3d", fig=None):
    if fig is None:
        fig = plt.gcf()
    rows, cols, start, stop = axi.get_subplotspec().get_geometry()
    ax.flat[start].remove()
    ax.flat[start] = fig.add_subplot(rows, cols, start + 1, projection=projection)


def normalize_max_d(max_d: dict) -> dict:
    config = OmegaConf.load(os.path.join(CONFIG_PATH, "sixd.yaml"))
    diffs = {i: config[i].max - config[i].min for i in config}
    mins = {i: config[i].min for i in config}
    print(diffs)
    print(mins)
    for i in diffs:
        for j in range(len(max_d[i])):
            max_d[i][j] = (max_d[i][j] - mins[i]) / diffs[i]

    slim_d = {}

    for i in mins:
        slim_d[i] = max_d[i]

    print(slim_d)
    # del slim_d["speed"]
    # del slim_d["pc"]
    # del slim_d["rmax"]

    return slim_d


def max_plots() -> None:
    line_styles = ["b-", "r-", "g-", "y--", "k--", "m--", "c--"]
    plot_defaults()
    fig, axs = plt.subplots(1, 2)  #  # subplot_kw=dict(projection="polar"))
    api_out = get_evolution("find-max-2")
    for i, exp in enumerate(api_out):
        axs[0].plot(
            [x + 1 for x in range(50)],
            api_out[exp]["metrics"][0]["values"],
            line_styles[i],
            label=f"{i+1}",
        )
    plt.legend()
    axs[0].set_xlim(1, 50)
    axs[0].set_xlabel("Number of Samples")
    axs[0].set_ylabel("Maximum Surge Height [m]")
    slim_d = normalize_max_d(get_max("find-max-2"))
    update_projection(axs, axs.flat[1], "polar")
    plot_slim_d(slim_d, axs.flat[1], line_styles)

    # ax.plot(angles, values, linewidth=1, linestyle="solid", label="group A")
    # ax.fill(angles, values, "b", alpha=0.1)

    # Add legend
    # plt.legend(
    #    loc="lower center", bbox_to_anchor=(0.5, -0.5), ncol=2
    # )  # loc="upper right", bbox_to_anchor=(0.1, 0.1))

    # ax.flat[i].set_title(pro_i)
    try:
        # axs[1].grid(True)
        pass
    except:
        pass
    label_subplots(axs)  # , y_pos=0.7, x_pos=0.05, fontsize=12)
    plt.savefig(os.path.join(FIGURE_PATH, "max_surge_group.png"))
    plt.clf()
    print(get_max("find-max-2"))


def plot_slim_d(slim_d: dict, ax: matplotlib.axes.Axes, line_styles: List[str]):
    # number of variable
    categories = list([key for key in slim_d])
    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot

    # If you want the first axis to be on top:
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], [SYMBOL_DICT[i] for i in categories])

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0, 0.5, 1], ["0", "0.5", "1"], color="grey", size=7)
    plt.ylim(-0.2, 1)

    # ------- PART 2: Add plots

    # Plot each individual = each line of the data
    # I don't make a loop, because plotting more than 3 groups makes the chart unreadable

    # Ind1
    # values = df.loc[0].drop("group").values.flatten().tolist()
    # values += values[:1]
    places_d = dict(
        ansley=27,
        new_orleans=5,
        diamondhead=17,
        mississippi=77,
        atchafayala=82,
        dulac=86,
        akers=2,
    )
    places_l = [key for key in places_d]
    for i in range(len(slim_d["xn"])):
        values = [slim_d[key][i] for key in slim_d]
        values += values[:1]
        ax.plot(angles, values, line_styles[i], linewidth=1, label=places_l[i])


# def ansley_plot()


if __name__ == "__main__":
    # python src/comet_load.py
    # ds_list = loop_through_experiment(METRICS)
    # plot_inum_metrics(ds_list)
    # plot_final_metrics(ds_list)
    # rose_plot()
    ansley_plot()
    max_plots()
    # loop_through_project()
    # loop_through_experiment()
    # python src/comet_load.py
