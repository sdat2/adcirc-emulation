"""Katrina."""
import os
import xarray as xr
import matplotlib.axes
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy
import cartopy.crs as ccrs
from sithom.time import timeit
from sithom.plot import plot_defaults, label_subplots
from src.preprocessing.sel import mid_katrina
from src.plot.map import add_features
from src.constants import (
    FIGURE_PATH,
    KATRINA_TIDE_NC,
    KATRINA_ERA5_NC,
    NEW_ORLEANS,
    NO_BBOX,
)


@timeit
def gauges_map(ax: matplotlib.axes.Axes) -> None:
    """
    Map of Tidal Gauges near New Orleans.

    Args:
        ax (matplotlib.axes.Axes):
    """
    ds = xr.open_dataset(KATRINA_TIDE_NC)

    ax.add_feature(cartopy.feature.LAND)
    add_features(ax)

    ax.set_extent(NO_BBOX.cartopy())

    for stationid in ds["stationid"].values:
        ax.plot(
            ds.sel(stationid=stationid).lon.values,
            ds.sel(stationid=stationid).lat.values,
            "X",
            label=stationid,
        )

    ax.plot(NEW_ORLEANS.lon, NEW_ORLEANS.lat, "X", color="black", label="New Orleans")
    ax.set_xticks(NO_BBOX.lon, crs=ccrs.PlateCarree())
    ax.set_yticks(NO_BBOX.lat, crs=ccrs.PlateCarree())
    ax.set_ylabel("Latitude [$^{\circ}$N]")
    ax.set_xlabel("Longitude [$^{\circ}$E]")
    plt.legend(title="stationid")


def era5(ax: matplotlib.axes.Axes, variable: str = "tp") -> None:
    """
    Add ERA5 from mid Katrina to the axes.

    Args:
        ax (matplotlib.axes.Axes): axes.
        variable (str, optional): Defaults to "tp".
    """
    pc = ccrs.PlateCarree()
    era5 = xr.open_dataset(KATRINA_ERA5_NC)
    mid_katrina(era5[variable]).plot(
        ax=ax, cmap="Greys", transform=pc
    )  # , #subplot_kws={'projection': pc})
    add_features(ax)


def tide_timeseries(ax: matplotlib.axes.Axes) -> None:
    """
    Tide time series.

    Args:
        ax (matplotlib.axes.Axes): tide time series.
    """
    ds = xr.open_dataset(KATRINA_TIDE_NC)
    ds.water_level.attrs["long_name"] = "Water level"
    ds.water_level.attrs["units"] = "m"
    # ds.water_level.attrs["units"] = "m"
    ds["date_time"].attrs["long_name"] = "Time"
    ds.water_level.plot.line(ax=ax, hue="stationid", alpha=0.7)


def triple_input_plot() -> None:
    """Triple input plot."""
    plot_defaults()
    fig = plt.figure(figsize=(8, 14))
    gs = fig.add_gridspec(3, 3)
    pc = ccrs.PlateCarree()
    ax3 = fig.add_subplot(gs[0, :], projection=pc)
    era5(ax3)

    ax1 = fig.add_subplot(gs[1, :], projection=pc)
    gauges_map(ax1)
    # ax1.coastlines(resolution='auto', color='k')
    # ax1.gridlines(color='lightgrey', linestyle='-', draw_labels=True)

    ax2 = fig.add_subplot(gs[2, :])
    # ax2.plot([1, 2], [3, 4])

    tide_timeseries(ax2)
    label_subplots([ax3, ax1, ax2], override="outside")
    plt.savefig(os.path.join(FIGURE_PATH, "inputs.pdf"))
    plt.savefig(os.path.join(FIGURE_PATH, "inputs.png"))


if __name__ == "__main__":
    # python src/plot/katrina_input.py
    triple_input_plot()
