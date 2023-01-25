"""Animate py."""
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import imageio

try:
    import cartopy
    import cartopy.crs as ccrs
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
except ImportError:
    print("cartopy not installed")
from sithom.place import BoundingBox
from sithom.plot import lim, plot_defaults
from sithom.time import timeit
from sithom.xr import plot_units
from src.constants import FIGURE_PATH, DATA_PATH, NO_BBOX, NEW_ORLEANS
from src.data_loading.adcirc import timeseries_height_ds, read_windspeeds


def animate_height_timeseries(ds: xr.Dataset, output_path: str) -> None:
    """
    Animate height timeseries.

    Args:
        ds (xr.Dataset): xarray dataset timeseries with mesh.
        output_path (str): output.
    """
    output_path = os.path.join(FIGURE_PATH, output_path)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    plot_defaults()
    vmin, vmax = lim(ds.zeta.values, percentile=0, balance=True)
    vmin, vmax = np.min([-vmax, vmin]), np.max([-vmin, vmax])
    levels = np.linspace(vmin, vmax, num=400)
    cbar_levels = np.linspace(vmin, vmax, num=7)

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
        plt.title(ts.strftime("%Y-%m-%d  %H:%M"))
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


def plot_quiver_height(path_in: str = "mult1", num: int = 185) -> None:
    """
    Plot quiver height.

    Args:
        path_in (str, optional): name of data folder. Defaults to "mult1".
        num (int, optional): num. Defaults to 185.
    """
    path_in = os.path.join(DATA_PATH, path_in)
    ds = timeseries_height_ds(path=path_in, bbox=NO_BBOX)
    print(ds)
    vmin, vmax = lim(ds.zeta.values, percentile=0, balance=True)
    vmin, vmax = np.min([-vmax, vmin]), np.max([-vmin, vmax])
    levels = np.linspace(vmin, vmax, num=400)
    cbar_levels = np.linspace(vmin, vmax, num=5)

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
    ax = plt.gca()
    cbar = plt.colorbar(label="Height [m]")
    cbar.set_ticks(cbar_levels)
    cbar.set_ticklabels(["{:.2f}".format(x) for x in cbar_levels.tolist()])
    plt.xlabel("Longitude [$^{\circ}$E]")
    plt.ylabel("Latitude [$^{\circ}$N]")
    time = ds.isel(time=num).time.values
    ts = pd.to_datetime(str(time))
    # plt.savefig(os.path.join(output_path, str(num) + ".png"))
    # plt.clf()
    ds = read_windspeeds(os.path.join(path_in, "fort.222"))
    # _, ax = plt.subplots(1, 1)
    quiver = plot_units(
        ds.sel(time=time, method="nearest"), x_dim="lon", y_dim="lat"
    ).plot.quiver(
        ax=ax,
        x="lon",
        y="lat",
        u="U10",
        v="V10",
        add_guide=False,
    )
    # x_pos = 0.65
    # y_pos = -0.15
    x_pos = 0.95
    y_pos = -0.15
    _ = plt.quiverkey(
        quiver,
        # 1.08,
        x_pos,
        y_pos,  # 08,
        40,
        str(r"$40$ m s$^{-1}$"),  # + "\n"
        labelpos="E",
        coordinates="axes"
        # coordinates="figure"
        # ,
    )
    NO_BBOX.ax_lim(plt.gca())
    plt.title(ts.strftime("%Y-%m-%d  %H:%M"))
    plt.savefig(os.path.join(FIGURE_PATH, "example_colision.png"))
    plt.clf()


@timeit
def animate_quiver_height(
    path_in: str = "mult1",
    output_path: str = "katrina_hit",
    bbox: BoundingBox = NO_BBOX,
) -> None:
    """
    Animate windspeed and sea surface height.

    Args:
        path_in (str, optional): folder name for input. Defaults to "mult1".
        output_path (str, optional): output folder name for images. Defaults to "katrina_hit".
        bbox (BoundingBox, optional): Bounding box. Defaults to NO_BBOX.
    """
    plot_defaults()
    output_path = os.path.join(FIGURE_PATH, output_path)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    path_in = os.path.join(DATA_PATH, path_in)
    ds = timeseries_height_ds(path=path_in, bbox=bbox.pad(buffer=2)).sel(
        time=slice("2005-08-28", "2005-08-30")
    )
    dsw = read_windspeeds(os.path.join(path_in, "fort.222"))
    print(ds)
    vmin, vmax = lim(ds.zeta.values, percentile=0, balance=True)
    vmin, vmax = np.min([-vmax, vmin]), np.max([-vmin, vmax])
    levels = np.linspace(vmin, vmax, num=400)
    cbar_levels = np.linspace(vmin, vmax, num=5)

    def plot_part(num=185):
        ax = plt.axes(projection=ccrs.PlateCarree())
        # maybe add a green-yellow backgroud here
        ax.set_facecolor("#d1ffbd")
        ax.add_feature(cartopy.feature.LAKES, alpha=0.5)
        # ax.add_feature(
        #    cartopy.feature.BORDERS, color="grey", alpha=0.5, linewidth=0.5
        # )  # linestyle=":")
        # ax.add_feature(
        #    cartopy.feature.STATES, color="grey", alpha=0.5, linewidth=0.5
        # )  # linestyle=":")
        ax.add_feature(cartopy.feature.RIVERS, alpha=0.5)
        plt.plot(
            NEW_ORLEANS.lon,
            NEW_ORLEANS.lat,
            marker=".",
            markersize=4,
            color="purple",
        )
        plt.text(
            NEW_ORLEANS.lon - 0.35,
            NEW_ORLEANS.lat - 0.16,
            "New Orleans",
            fontsize=6,
            color="purple",
        )
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

        ax = plt.gca()
        cbar = plt.colorbar(label="Sea Surface Height, $\eta$ [m]")
        cbar.set_ticks(cbar_levels)
        cbar.set_ticklabels(["{:.2f}".format(x) for x in cbar_levels.tolist()])
        plt.xlabel("")
        plt.ylabel("")
        time = ds.isel(time=num).time.values
        ts = pd.to_datetime(str(time))
        # plt.savefig(os.path.join(output_path, str(num) + ".png"))
        # plt.clf()
        # _, ax = plt.subplots(1, 1)
        quiver = plot_units(
            dsw.sel(time=time, method="nearest"), x_dim="lon", y_dim="lat"
        ).plot.quiver(
            ax=ax,
            x="lon",
            y="lat",
            u="U10",
            v="V10",
            add_guide=False,
        )
        # x_pos = 0.65
        # y_pos = -0.15
        x_pos = 1.01
        y_pos = -0.10
        _ = plt.quiverkey(
            quiver,
            # 1.08,
            x_pos,
            y_pos,  # 08,
            40,
            str(r"$40$ m s$^{-1}$"),  # + "\n"
            labelpos="E",
            coordinates="axes",
            transform=ccrs.PlateCarree(),
            # coordinates="figure"
            # ,
        )
        ax.set_extent(bbox.cartopy(), crs=ccrs.PlateCarree())

        # ax.yaxis.tick_right()
        ax.set_yticks(
            [26, 27, 28, 29, 30, 31],
            # labels=[26, 27, 28, 29, 30, 31],
            crs=ccrs.PlateCarree(),
        )
        ax.set_xticks(
            [-93, -92, -91, -90, -89, -88],
            # labels=[-93, -92, -91, -90, -89, -88],
            crs=ccrs.PlateCarree(),
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        # lon_formatter = LongitudeFormatter(zero_direction_label=True)
        # lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
        ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)

        # bbox.ax_lim(plt.gca())
        # bbox.ax_labels(plt.gca())
        # cartopy.mpl.gridliner.Gridliner(ax, ccrs.PlateCarree())
        plt.title(ts.strftime("%Y-%m-%d  %H:%M"))
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


if __name__ == "__main__":
    plot_defaults()
    # python src/plot/animate.py
    # trim_and_animate("kate_h08", "katrina_hit5")
    # trim_and_animate("katd_h08", "katrina_hit5")
    # trim_and_animate("mult2", "katrina_mult2")
    # animate_quiver_height(
    #     path_in="mult1", output_path="katrina_hit_larger", bbox=NEW_ORLEANS.bbox(5)
    # )
    bbox = NEW_ORLEANS.bbox(3)
    bbox.lat = [x - 1.5 for x in bbox.lat]

    animate_quiver_height(
        path_in="kat_angle/b-53.636_kat_angle",
        output_path="katrina_hit_near_max",
        bbox=bbox,
    )
    animate_quiver_height(
        path_in="kat_move_smeared/x0.606_kat_move",
        output_path="katrina_hit_holland0.6_smeared",
        bbox=bbox,
    )

    # animate_quiver_hdeight(
    #    path_in="kat_move_smeared/x0.606_kat_move",
    #    output_path="katrina_hit_holland0.6_smeared",
    #   bbox=bbox,
    # )
