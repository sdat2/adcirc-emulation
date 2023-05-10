"""
Comet load.py
"""
import os
import math
import numpy as np
from typing import List, Tuple, Callable
import comet_ml
from comet_ml import API
import xarray as xr
import matplotlib.pyplot as plt
from src.constants import FIGURE_PATH, DATA_PATH
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
    plot_defaults()
    api_out = comet_api.get_metrics_for_chart(
        [exp.id for exp in comet_api.get("sdat2", "find-max-naive")],
        # ["angle", "speed", "point_east", "rmax", "pc", "xn", "max", "step"],
        ["max"],
        ["init_samples", "active_samples", "seed"],
    )
    for i, exp in enumerate(api_out):
        plt.plot(api_out[exp]["metrics"][0]["values"], label=f"a{i+1}")
    api_out = comet_api.get_metrics_for_chart(
        [exp.id for exp in comet_api.get("sdat2", "find-max-full-active")],
        # ["angle", "speed", "point_east", "rmax", "pc", "xn", "max", "step"],
        ["max"],
        ["init_samples", "active_samples", "seed"],
    )
    for i, exp in enumerate(api_out):
        plt.plot(api_out[exp]["metrics"][0]["values"], "--", label=f"b{i+1}")
    plt.legend()
    plt.xlabel("Number of Samples")
    plt.ylabel("Maximum Surge Height [m]")
    plt.savefig(os.path.join(FIGURE_PATH, "max_surge_ansley.png"))


def max_plots() -> None:
    plot_defaults()
    api_out = comet_api.get_metrics_for_chart(
        [exp.id for exp in comet_api.get("sdat2", "find-max-2")],
        # ["angle", "speed", "point_east", "rmax", "pc", "xn", "max", "step"],
        ["max"],
        ["init_samples", "active_samples", "seed"],
    )
    for i, exp in enumerate(api_out):
        plt.plot(api_out[exp]["metrics"][0]["values"], label=f"a{i+1}")
    plt.legend()
    plt.xlabel("Number of Samples")
    plt.ylabel("Maximum Surge Height [m]")
    plt.savefig(os.path.join(FIGURE_PATH, "max_surge_.png"))


# def ansley_plot()


if __name__ == "__main__":
    # python src/comet_load.py
    # ds_list = loop_through_experiment(METRICS)
    # plot_inum_metrics(ds_list)
    # plot_final_metrics(ds_list)
    # rose_plot()
    ansley_plot()
    # loop_through_project()
    # loop_through_experiment()
    # python src/comet_load.py
