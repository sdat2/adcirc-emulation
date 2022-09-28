"""Plot hurricane generation."""
import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sithom.misc import in_notebook
from sithom.plot import plot_defaults
from src.models.generation import HollandTropicalCyclone
from src.plot.map import map_axes
from src.constants import NEW_ORLEANS, FIGURE_PATH


def different_trajectories() -> None:
    """
    Plot different trajectories for the idealised hurricane models.
    """
    #    plot_katrina_windfield_example(model=key)
    plot_defaults()
    ax = map_axes()

    for angle in [-60, 50, -40, -30, -15, 0, 15, 30, 40, 50, 60, 70]:
        htc = HollandTropicalCyclone(NEW_ORLEANS, angle, 3, 50, 10e3, 100)
        traj_ds = htc.trajectory_ds()
        print(traj_ds)
        plt.plot(traj_ds.lon.values, traj_ds.lat.values, alpha=0.5)
        print([x for x in traj_ds.time.values.tolist()])
        print(traj_ds.time.values)
        # times = np.array([np.datetime64(x) for x in traj_ds.time.values.tolist()])
        im = plt.scatter(
            traj_ds.lon.values,
            traj_ds.lat.values,
            c=mdates.date2num(traj_ds.time.values),
            s=1,
        )
        plt.scatter(
            htc.point.lon, htc.point.lat,  # c=dates
        )

    # plt.colorbar(im, label="Time")

    ylabel = r"Latitude [$^{\circ}$N]"
    xlabel = r"Longitude [$^{\circ}$E]"

    cb = plt.colorbar()
    loc = mdates.AutoDateLocator()
    cb.ax.yaxis.set_major_locator(loc)
    cb.ax.yaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))

    ax.set(xlabel=xlabel, ylabel=ylabel)
    ylocs = [20, 24, 28, 32, 36]
    xlocs = [-97, -93, -89, -85, -81]
    ax.gridlines(draw_labels=True, ylocs=ylocs, xlocs=xlocs)

    if in_notebook():
        plt.show()
    else:
        plt.tight_layout()
        plt.savefig(
            os.path.join(FIGURE_PATH, "example_trajectories.png"), bbox_inches="tight"
        )
        plt.clf()


if __name__ == "__main__":
    different_trajectories()
    # python src/plot/generation.py
