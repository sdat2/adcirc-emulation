"""IBTRACS plotting."""
import os
import matplotlib.axes
import matplotlib.pyplot as plt
import xarray as xr
from sithom.misc import in_notebook
from sithom.time import timeit
from sithom.plot import plot_defaults
from src.constants import FIGURE_PATH, GOM_BBOX, NO_BBOX
from src.data_loading.ibtracs import na_tcs, gom_tcs
from src.plot.map import map_axes


def plot_storm(
    ax: matplotlib.axes.Axes,
    ibtracs_ds: xr.Dataset,
    var: str = "storm_speed",
    storm_num: int = 0,
    cmap="viridis",
    scatter_size: float = 1.6,
) -> None:
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
    ax.scatter(
        ibtracs_ds[var].isel(storm=storm_num)["lon"].values,
        ibtracs_ds[var].isel(storm=storm_num)["lat"].values,
        c=ibtracs_ds[var].isel(storm=storm_num).values,
        marker=".",
        s=scatter_size,
        cmap=cmap,
    )


@timeit
def plot_multiple_storms(
    ibtracs_ds: xr.Dataset, ax: matplotlib.axes.Axes=None, var="storm_speed", cmap="viridis", scatter_size: float = 1.6,
) -> None:
    """
    Plot all the storms in an IBTRACS dataset.

    Args:
        ibtracs_ds (xr.Dataset): IBTRACS dataset.
        var (str, optional): Variable to plot. Defaults to "storm_speed".
        cmap (str, optional): Colormap. Defaults to "viridis".
    """
    ax = map_axes()
    print(ibtracs_ds)
    for num in range(0, ibtracs_ds.storm.shape[0]):
        plot_storm(
            ax, ibtracs_ds, var=var, storm_num=num, cmap=cmap, scatter_size=scatter_size
        )

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


if __name__ == "__main__":
    # python src/plot/ibtracs.py
    plot_defaults()
    plot_na_tcs()
    plot_gom_tcs()
    print(GOM_BBOX)
    print(NO_BBOX)
