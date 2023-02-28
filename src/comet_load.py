"""
Comet load.py
"""
import os
import numpy as np
from typing import List
import comet_ml
from comet_ml import API
import xarray as xr
import matplotlib.pyplot as plt
from src.constants import FIGURE_PATH
from sithom.plot import plot_defaults, label_subplots

comet_ml.config.save(api_key="57fHMWwvxUw6bvnjWLvRwSQFp")

comet_api = API()
print(comet_api.get())


def loop_through_project():
    workspace = "sdat2"  # /6dactive"
    project = "6dactive"

    for exp in comet_api.get(workspace, project):
        print("    processing project", project, "...")
        print("        processing experiment", exp.id, end="")
        print(".", end="")
        print(comet_api.get(workspace, project, exp.id))


def loop_through_experiment() -> List[xr.Dataset]:
    workspace = "sdat2"  # /6dactive"
    project = "6dactive"
    # experiment = "f5e7f5b6d8c34f9b9c3d7e5a6d2f2c3"
    # experiment = "261f1786b8ab496e90170d593deba88f"
    ds_list = []
    for exp in comet_api.get(workspace, project):

        # exp = comet_api.get(workspace, project, experiment)
        metric_d = {}
        for metric, typ in [
            ("inum", int),
            ("anum", int),
            ("mae", float),
            ("rmse", float),
            ("r2", float),
        ]:
            metrics = exp.get_metrics(metric)
            metric_l = [typ(metrics[i]["metricValue"]) for i in range(len(metrics))]
            metric_d[metric] = (["point"], metric_l)
            print(exp.id, metric, "len(metrics)", len(metrics))
            # print("metrics", metrics)
            # for i in range(len(metrics)):
            #    print("metrics[" + str(i) + "]", metrics[i]["metricValue"])
        # print("metrics[0]", metrics[0]["metricValue"])
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
        max_i = max(max_i, ds.inum.max().values)
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
        max_i = max(max_i, ds.inum.max().values)

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
        inum_l.append(ds.inum.values[-1])
        anum_l.append(ds.anum.values[-1])
        r2_l.append(ds.r2.values[-1])
        mae_l.append(ds.mae.values[-1])
        rmse_l.append(ds.rmse.values[-1])
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


ds_list = loop_through_experiment()
plot_inum_metrics(ds_list)
plot_final_metrics(ds_list)
# loop_through_project()
# loop_through_experiment()
# python src/comet_load.py
