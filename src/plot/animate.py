"""Animate py."""
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import imageio
from sithom.place import BoundingBox
from sithom.plot import lim, plot_defaults
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


def prcp_quiver_plot(
    show_plots: bool = False,
    save_path=str(os.path.join(FIGURE_PATH, "uv_prcp.png")),
) -> None:
    """
    prcp wind trends quiver plot.
    Args:
        setup (ModelSetup, optional): setup object. Defaults to get_default_setup().
        show_plots (bool, optional): Whether to show plots. Defaults to False.
        save_path ([type], optional): Path to save fig to. Defaults to
            str(FIGURE_PATH / "uv_prcp.png").
    """

    ds = read_windspeeds(os.path.join(DATA_PATH, path_in, "fort.222"))
    _, ax = plt.subplots(1, figsize=get_dim(ratio=0.5))
    ads = xr.open_dataset(setup.tcam_output())
    pqp_part(ax, ads)
    # plt.title("$\Delta$ precipitation and wind velocities [m s$^{-1}$]")
    plt.tight_layout()
    plt.savefig(save_path)
    if show_plots:
        plt.show()
    else:
        plt.clf()


def pqp_part(
    ax: matplotlib.axes.Axes, ads: xr.Dataset, x_pos=0.65, y_pos=-0.15, **kwargs
) -> None:
    """
    Plot a panel of pqp figure.
    Args:
        ax (matplotlib.axes.Axes): axes to plot on.
        ads (xr.Dataset): Standard atmos dataset.
    """
    new_x = list(range(-100, 291, 5))
    new_y = list(range(-30, 31, 5))
    fvtrend = interp2d(ads.X, ads.Yv, ads.vtrend, kind="linear")
    futrend = interp2d(ads.X, ads.Yu, ads.utrend, kind="linear")
    new_ds = xr.Dataset(
        {
            "X": ("X", new_x),
            "Y": ("Y", new_y),
        }
    )
    new_ds.X.attrs = [("units", "degree_east")]
    new_ds.Y.attrs = [("units", "degree_north")]
    new_ds["utrend"] = (["Y", "X"], futrend(new_x, new_y))
    new_ds["vtrend"] = (["Y", "X"], fvtrend(new_x, new_y))
    clip(can_coords(ads.PRtrend)).plot(
        ax=ax,
        cmap=cmap("ranom"),
        cbar_kwargs={"label": "Precipitation [m s$^{-1}$]"},
        **kwargs
    )
    quiver = add_units(new_ds).plot.quiver(
        ax=ax, x="X", y="Y", u="utrend", v="vtrend"
    )  # , normalize=matplotlib.colors.Normalize(vmin=0.01, vmax=1))#, scale=30)
    _ = plt.quiverkey(
        quiver,
        # 1.08,
        x_pos,
        y_pos,  # 08,
        1,
        str(r"$1$ m s$^{-1}$" + r" $\Delta \vec{u}$"),  # + "\n"
        labelpos="E",
        coordinates="axes"
        # coordinates="figure"
    )


def plot_single_quiver() -> None:
    ds = read_windspeeds(os.path.join(DATA_PATH, "mult2", "fort.222"))
    print(ds.U10, ds.V10)

    _, ax = plt.subplots(1, 1)
    quiver = plot_units(ds.isel(time=40), x_dim="lon", y_dim="lat").plot.quiver(
        ax=ax,
        x="lon",
        y="lat",
        u="U10",
        v="V10",
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


def plot_quiver_height(path_in: str = "mult1", num: int = 185) -> None:
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
    ds = timeseries_height_ds(path=path_in, bbox=bbox).sel(
        time=slice("2005-08-28", "2005-08-30")
    )
    dsw = read_windspeeds(os.path.join(path_in, "fort.222"))
    print(ds)
    vmin, vmax = lim(ds.zeta.values, percentile=0, balance=True)
    vmin, vmax = np.min([-vmax, vmin]), np.max([-vmin, vmax])
    levels = np.linspace(vmin, vmax, num=400)
    cbar_levels = np.linspace(vmin, vmax, num=5)

    def plot_part(num=185):
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
        cbar = plt.colorbar(label="Sea Surface Height [m]")
        cbar.set_ticks(cbar_levels)
        cbar.set_ticklabels(["{:.2f}".format(x) for x in cbar_levels.tolist()])
        plt.xlabel("Longitude [$^{\circ}$E]")
        plt.ylabel("Latitude [$^{\circ}$N]")
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
        bbox.ax_lim(plt.gca())
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

    # animate_quiver_height(
    #    path_in="kat_move_smeared/x0.606_kat_move",
    #    output_path="katrina_hit_holland0.6_smeared",
    #   bbox=bbox,
    # )
