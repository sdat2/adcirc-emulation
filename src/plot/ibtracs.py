"""IBTRACS plotting."""
from typing import Optional, Tuple, List
import os
import numpy as np
import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import matplotlib.path as mpath
import xarray as xr
from sithom.misc import in_notebook
from sithom.time import timeit
from sithom.plot import plot_defaults, lim
from sithom.place import BoundingBox
from sithom.plot import label_subplots, set_dim
from src.constants import FIGURE_PATH, GOM_BBOX, NO_BBOX
from src.data_loading.ibtracs import na_tcs, gom_tcs, landing_distribution
from src.plot.map import map_axes
from src.preprocessing.labels import sanitize


def plot_storm(
    ax: matplotlib.axes.Axes,
    ibtracs_ds: xr.Dataset,
    var: str = "storm_speed",
    storm_num: int = 0,
    cmap: str = "viridis",
    scatter_size: float = 1.6,
    vmin: Optional[Tuple[float]] = None,
    vmax: Optional[Tuple[float]] = None,
) -> any:
    """
    Plot storm.

    Args:
        ibtracs_ds (xr.Dataset): IBTRACS dataset.
        var (str, optional): Variable to plot. Defaults to "storm_speed".
        storm_num (int, optional): Which storm to plot. Defaults to 0.
        cmap (str, optional): Which cmap to use. Defaults to "viridis".
        scatter_size (float, optional): Defaults to 1.6.
        vmin (Optional[Tuple[float]], optional): Defaults to None.
        vmax (Optional[Tuple[float]], optional): Defaults to None.

    any:

    """
    ax.plot(
        ibtracs_ds[var].isel(storm=storm_num)["lon"].values,
        ibtracs_ds[var].isel(storm=storm_num)["lat"].values,
        color="black",
        linewidth=0.1,
        alpha=0.5,
    )
    if vmin is not None and vmax is not None:
        im = ax.scatter(
            ibtracs_ds[var].isel(storm=storm_num)["lon"].values,
            ibtracs_ds[var].isel(storm=storm_num)["lat"].values,
            c=ibtracs_ds[var].isel(storm=storm_num).values,
            marker=".",
            s=scatter_size,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
    else:
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
    var: str = "storm_speed",
    cmap: str = "viridis",
    scatter_size: float = 1.6,
    bbox: Optional[BoundingBox] = None,
) -> None:
    """
    Plot all the storms in an IBTRACS dataset.

    Args:
        ibtracs_ds (xr.Dataset): IBTRACS dataset.
        var (str, optional): Variable to plot. Defaults to "storm_speed".
        cmap (str, optional): Colormap. Defaults to "viridis".
        scatter_size (float, optional): Scatter size. Defaults to 1.6.
        bbox (Optional[BoundingBox], optional): `BoundingBox` for plot. Defaults to None.
    """
    if ax is None:
        ax = map_axes()

    vlim = lim(ibtracs_ds[var].values)
    for num in range(0, ibtracs_ds.storm.shape[0]):
        im = plot_storm(
            ax,
            ibtracs_ds,
            var=var,
            storm_num=num,
            cmap=cmap,
            scatter_size=scatter_size,
            vmin=vlim[0],
            vmax=vlim[1],
        )
    if bbox is not None:
        bbox.ax_lim(ax)
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # plt.colorbar(im, cax=cax)
    ax.set_ylabel(r"Latitude [$^{\circ}$N]")
    ax.set_xlabel(r"Longitude [$^{\circ}$E]")
    # cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
    plt.gcf().colorbar(
        im,
        label=str(sanitize(var) + " [" + ibtracs_ds[var].attrs["units"] + "]"),
        fraction=0.046,
        pad=0.04,
    )


@timeit
def plot_na_tcs(var="storm_speed", scatter_size: float = 1.6) -> None:
    """
    Plot NA Tropical Cyclones.
    """
    # plot_defaults()
    plot_multiple_storms(na_tcs(), var=var, scatter_size=scatter_size)
    plt.savefig(os.path.join(FIGURE_PATH, "na_tc_speed.png"))
    if not in_notebook():
        plt.clf()


@timeit
def plot_gom_tcs(var="storm_speed", scatter_size: float = 1.6) -> None:
    """
    Plot GOM Tropical Cyclones.
    """
    # plot_defaults()
    plot_multiple_storms(gom_tcs(), var=var, scatter_size=scatter_size)
    plt.savefig(os.path.join(FIGURE_PATH, "gom_tc_speed.png"))
    if not in_notebook():
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
    if not in_notebook():
        plt.clf()


def var_label(ds: xr.Dataset, var: str) -> str:
    """
    Var label.

    Args:
        ds (xr.Dataset): dataset.
        var (str): variable string.

    Returns:
        str: Label.
    """
    return ds[var].attrs["long_name"] + " [" + ds[var].attrs["units"] + "]"


def plain_hist(
    dist: np.ndarray,
    ds: xr.Dataset,
    var: str,
    ax: Optional[matplotlib.axes.Axes] = None,
) -> None:
    """
    Plain Histogram.

    Args:
        dist (np.ndarray): distribution array.
        ds (xr.Dataset): _description_
        var (str): _description_
        ax (Optional[matplotlib.axes.Axes], optional): _description_. Defaults to None.
    """
    if ax is None:
        ax = plt.subplot(projection="polar")
    if var == "usa_sshs":
        kwargs = dict(bins=[0.5 + x for x in range(6)])
    else:
        kwargs = dict()
    ax.hist(dist, **kwargs)
    ax.set_ylabel("Number of landings")
    ax.set_xlabel(var_label(ds, var))


def angle_hist(
    dist: np.ndarray,
    ds: xr.Dataset,
    var: str,
    ax: Optional[matplotlib.axes.Axes] = None,
) -> None:
    """
    Plot polar histogram.

    Will only work for single plot,
    as uses plt.hist to create histogram bins.

    Args:
        dist (np.ndarray): Input array [Degrees].
        ds (xr.Dataset): Dataset.
        var (str): variable.
        ax (Optional[matplotlib.axes.Axes]): axes.
    """
    output = plt.hist(dist)
    points = output[0]
    rads = output[1] / 360 * 2 * np.pi
    plt.clf()
    if ax is None:
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


def individual_dist(
    ds: xr.Dataset, var: str, ax: Optional[matplotlib.axes.Axes] = None
) -> None:
    """
    Individual distribution.

    Args:
        ds (xr.Dataset): _description_
        var (str): _description_
        ax (Optional[matplotlib.axes.Axes], optional): _description_. Defaults to None.
    """
    dist = landing_distribution(ds, var=var)
    if var == "x":  # "storm_dir":
        angle_hist(dist, ds, var, ax=ax)
    else:
        plain_hist(dist, ds, var, ax=ax)
    if var == "usa_sshs":
        kwargs = dict(bins=[0.5 + x for x in range(6)])


def multi_dist(ds: xr.Dataset, var_list: List[List[str]]) -> None:
    """
    Multi distibution.

    Args:
        ds (xr.Dataset): IBTRaCS dataset.
        var_list (List[List[str]]): IBTRaCS variables to plot in requireed shape.
    """
    var_array = np.array(var_list)
    shape = var_array.shape
    fig, axs = plt.subplots(*shape)
    set_dim(fig, fraction_of_line_width=1.5, ratio=(5**0.5 - 1) / 2 * 2)

    for i in range(shape[0]):
        for j in range(shape[1]):
            individual_dist(ds, var_array[i, j], ax=axs[i, j])
    if len(axs.ravel()) > 1:
        label_subplots(axs.ravel(), override="outside")


def make_multi_dist() -> None:
    """
    Make multi-panel distribution plot.
    """
    gtcs = gom_tcs()
    multi_dist(
        gtcs,
        [
            ["storm_speed", "storm_dir"],
            ["usa_pres", "usa_rmw"],
            ["usa_wind", "usa_sshs"],
        ],
    )
    if in_notebook():
        plt.show()
    else:
        plt.savefig(os.path.join(FIGURE_PATH, "gom_landing_distributions.png"))
        plt.clf()


if __name__ == "__main__":
    # python src/plot/ibtracs.py
    plot_defaults()
    # plot_na_tcs()
    # plot_gom_tc_angles()
    # plot_gom_tcs()
    print(GOM_BBOX)
    print(NO_BBOX)
    make_multi_dist()
