"""Animate py."""
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import imageio
from sithom.plot import lim, plot_defaults
from src.constants import FIGURE_PATH, DATA_PATH
from src.data_loading.adcirc import timeseries_height_ds

plot_defaults()


def animate_height_timeseries(ds: xr.Dataset, output_path: str) -> None:
    """
    Animate height timeseries.

    Args:
        ds (xr.Dataset): xarray dataset timeseries with mesh.
        output_path (str): output.
    """
    output_path = os.path.join(FIGURE_PATH, output_path)
    plot_defaults()
    vmin, vmax = lim(ds.zeta.values, percentile=0, balance=True)
    vmin, vmax = np.min([-vmax, vmin]), np.max([-vmin, vmax])
    levels = np.linspace(vmin, vmax, num=400)
    cbar_levels = np.linspace(vmin, vmax, num=7)

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    def plot_part(num: int = 125) -> None:
        plt.tricontourf(
            ds.lon.values,
            ds.lat.values,
            ds.mesh.values,
            ds.zeta.values[num],
            vmin=vmin,
            vmax=vmax,
            levels=levels,
            cmap="cmo.balance",
        )
        cbar = plt.colorbar(label="Height [m]")
        cbar.set_ticks(cbar_levels)
        cbar.set_ticklabels(["{:.2f}".format(x) for x in cbar_levels.tolist()])

        plt.xlabel("Longitude [$^{\circ}$E]")
        plt.ylabel("Latitude [$^{\circ}$N]")
        time = ds.isel(time=num).time.values
        ts = pd.to_datetime(str(time))
        plt.title(ts.strftime("%Y-%m-%d %H:%M"))
        plt.savefig(os.path.join(output_path, str(num) + ".png"))
        plt.clf()

    for i in range(len(ds.time.values)):
        plot_part(i)

    fig_paths = [
        os.path.join(output_path, str(num) + ".png")
        for num in range(len(ds.time.values))
    ]
    ims = [imageio.imread(f) for f in fig_paths]
    imageio.mimwrite(output_path + ".gif", ims)


def trim_and_animate(path_in: str, path_out: str) -> None:
    """
    Trim and animate.

    Args:
        path_in (str): path in.
        path_out (str): path out.
    """
    path_in = os.path.join(DATA_PATH, path_in)
    ds = timeseries_height_ds(path=path_in)
    animate_height_timeseries(ds.sel(time=slice("2005-08-27", "2005-08-31")), path_out)


if __name__ == "__main__":
    trim_and_animate("kate_h08", "katrina_hit5")
