"""IBTRACS plotting."""
from typing import Optional
import os
import numpy as np
import matplotlib.axes
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import xarray as xr
from sithom.misc import in_notebook
from sithom.time import timeit
from sithom.plot import plot_defaults
from src.constants import FIGURE_PATH, GOM_BBOX, NO_BBOX
from src.data_loading.ibtracs import na_tcs, gom_tcs
from src.place import BoundingBox
from src.plot.map import map_axes


def plot_storm(
    ax: matplotlib.axes.Axes,
    ibtracs_ds: xr.Dataset,
    var: str = "storm_speed",
    storm_num: int = 0,
    cmap: str ="viridis",
    scatter_size: float = 1.6,
) -> any:
    """
    Plot storm.

    Args:
        ibtracs_ds (xr.Dataset): IBTRACS dataset.
        var (str, optional): Variable to plot. Defaults to "storm_speed".
        storm_num (int, optional): Which storm to plot. Defaults to 0.
        cmap (str, optional): Which cmap to use. Defaults to "viridis".
    """
    ax.plot(
        ibtracs_ds[var].isel(storm=storm_num)["lon"].values,
        ibtracs_ds[var].isel(storm=storm_num)["lat"].values,
        color="black",
        linewidth=0.1,
        alpha=0.5,
    )
    im = ax.scatter(
        ibtracs_ds[var].isel(storm=storm_num)["lon"].values,
        ibtracs_ds[var].isel(storm=storm_num)["lat"].values,
        c=ibtracs_ds[var].isel(storm=storm_num).values,
        marker=".",
        s=scatter_size,
        cmap=cmap,
    )
    return im


@timeit
def plot_multiple_storms(
    ibtracs_ds: xr.Dataset,
    ax: matplotlib.axes.Axes = None,
    var="storm_speed",
    cmap="viridis",
    scatter_size: float = 1.6,
    bbox: Optional[BoundingBox] = None
) -> None:
    """
    Plot all the storms in an IBTRACS dataset.

    Args:
        ibtracs_ds (xr.Dataset): IBTRACS dataset.
        var (str, optional): Variable to plot. Defaults to "storm_speed".
        cmap (str, optional): Colormap. Defaults to "viridis".
        scatter_size (float, optional): Scatter size. Defaults to 1.6
        bbox (Optional[BoundingBox], optional): `BoundingBox` for plot. Defaults to None.
    """
    if ax is None:
        ax = map_axes()
    for num in range(0, ibtracs_ds.storm.shape[0]):
        im = plot_storm(
            ax, ibtracs_ds, var=var, storm_num=num, cmap=cmap, scatter_size=scatter_size
        )
    if bbox is not None:
        bbox.ax_lim(ax)
    #divider = make_axes_locatable(ax)
    #cax = divider.append_axes('right', size='5%', pad=0.05)
    #plt.colorbar(im, cax=cax)
    ax.set_ylabel(r"Latitude [$^{\circ}$N]")
    ax.set_xlabel(r"Longitude [$^{\circ}$E]")
    # ax.colorbar(label=var + " [" + ibtracs_ds[var].attrs["units"] + "]")


@timeit
def plot_na_tcs(var="storm_speed", scatter_size: float = 1.6) -> None:
    """
    Plot NA Tropical Cyclones.
    """
    # plot_defaults()
    plot_multiple_storms(na_tcs(), var=var, scatter_size=scatter_size)
    plt.savefig(os.path.join(FIGURE_PATH, "na_tc_speed.png"))
    if not in_notebook:
        plt.clf()


@timeit
def plot_gom_tcs(var="storm_speed", scatter_size: float = 1.6) -> None:
    """
    Plot GOM Tropical Cyclones.
    """
    # plot_defaults()
    plot_multiple_storms(gom_tcs(), var=var, scatter_size=scatter_size)
    plt.savefig(os.path.join(FIGURE_PATH, "gom_tc_speed.png"))
    if not in_notebook:
        plt.clf()


def polar_hist(input: np.ndarray) -> None:
    """
    Plot polar histogram.

    Will only work for single plot,
    as uses plt.hist to create histogram bins.

    Args:
        input (np.ndarray): Input array [Degrees].
    """
    output = plt.hist(input)
    points = output[0]
    rads = output[1] / 360 * 2 * np.pi
    plt.clf()
    ax = plt.subplot(projection="polar")
    ax.bar(
        rads[1:],
        points,
        width=2 * np.pi / len(points),
        bottom=0.0,
        alpha=0.5,
        edgecolor="black",
    )
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    # plt.show()


def plot_gom_tc_angles() -> None:
    """
    Plot gom tc angles.
    """
    polar_hist(gom_tcs().storm_dir.values.ravel())
    plt.savefig(os.path.join(FIGURE_PATH, "gom_tc_angles.png"))
    if not in_notebook:
        plt.clf()


if __name__ == "__main__":
    # python src/plot/ibtracs.py
    plot_defaults()
    plot_na_tcs()
    plot_gom_tc_angles()
    plot_gom_tcs()
    print(GOM_BBOX)
    print(NO_BBOX)
