"""IBTRACS plotting."""
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from sithom.plot import plot_defaults
from src.data_loading.ibtracs import na_tcs


def plot_storm(ds: xr.Dataset, var: str ="storm_speed", storm_num: int = 0, cmap="viridis") -> None:
    """
    Plot storm.

    Args:
        ds (xr.Dataset): IBTRACS dataset.
        var (str, optional): Variable to plot. Defaults to "storm_speed".
        storm_num (int, optional): Which storm to plot. Defaults to 0.
        cmap (str, optional): Which cmap to use. Defaults to "viridis".
    """
    plt.scatter(
        ds[var].isel(storm=storm_num)["lon"].values,
        ds[var].isel(storm=storm_num)["lat"].values,
        c=ds[var].isel(storm=storm_num).values,
        marker=".",
        cmap=cmap,
    )

def plot_all_storms(ds: xr.Dataset, var="storm_speed", cmap="viridis") -> None:
    """
    Plot all the storms in an IBTRACS dataset.

    Args:
        ds (xr.Dataset): IBTRACS dataset.
        var (str, optional): Variable to plot. Defaults to "storm_speed".
        cmap (str, optional): Colormap. Defaults to "viridis".
    """
    for num in range(0, ds.storm.shape[0]):
        plot_storm(ds, var=var, storm_num=num, cmap=cmap)

    plt.ylabel(r"Latitude [$^{\circ}N$]")
    plt.xlabel(r"Longitude [$^{\circ}E$]")


def plot_na_tcs() -> None:
    """
    Plot NA tropical cyclones.
    """
    plot_defaults()
    plot_all_storms(na_tcs())
    plt.savefig("example.png")


if __name__ == "__main__":
    # python src/plot/ibtracs.py
    plot_na_tcs()
