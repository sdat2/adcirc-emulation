"""IBTRACS plotting."""
from typing import Optional, Tuple
import os
import numpy as np
import matplotlib.axes
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import matplotlib.path as mpath
import xarray as xr
from sithom.misc import in_notebook
from sithom.time import timeit
from sithom.plot import plot_defaults, lim
from sithom.place import BoundingBox
from src.constants import FIGURE_PATH, GOM_BBOX, NO_BBOX
from src.data_loading.ibtracs import na_tcs, gom_tcs
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


def colorline(
    x: np.ndarray,
    y: np.ndarray,
    z: Optional[np.ndarray] = None,
    cmap=plt.get_cmap("copper"),
    norm=plt.Normalize(0.0, 1.0),
    linewidth: float = 3,
    alpha: float = 1.0,
) -> mcoll.LineCollection:
    """
    https://stackoverflow.com/a/25941474

    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(
        segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha
    )

    ax = plt.gca()
    ax.add_collection(lc)

    return lc


def make_segments(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    https://stackoverflow.com/a/25941474

    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def test_colorline() -> None:
    """
    https://stackoverflow.com/a/25941474
    """
    N = 10
    np.random.seed(101)
    x = np.random.rand(N)
    y = np.random.rand(N)
    fig, ax = plt.subplots()
    path = mpath.Path(np.column_stack([x, y]))
    verts = path.interpolated(steps=3).vertices
    x, y = verts[:, 0], verts[:, 1]
    z = np.linspace(0, 1, len(x))
    colorline(x, y, z, cmap=plt.get_cmap("jet"), linewidth=2)
    plt.show()


if __name__ == "__main__":
    # python src/plot/ibtracs.py
    plot_defaults()
    plot_na_tcs()
    plot_gom_tc_angles()
    plot_gom_tcs()
    print(GOM_BBOX)
    print(NO_BBOX)
