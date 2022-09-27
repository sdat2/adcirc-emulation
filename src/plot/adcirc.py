"""ADCIRC plot and animate."""
from typing import Callable
import os
from tqdm import tqdm
import numpy as np
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import adcircpy
from adcircpy.outputs import (
    Maxele,
    MaximumElevationTimes,
    Fort63,
    Fort61,
    Minpr,
    Maxwvel,
    Maxvel,
)
import imageio
import cmocean.cm as cmo
from sithom.plot import label_subplots, plot_defaults, get_dim, lim, cmap
from sithom.xr import plot_units
from sithom.misc import in_notebook
from src.constants import DATA_PATH, FIGURE_PATH, KAT_EX_PATH


def maxele() -> None:
    """
    Max elevation.
    """
    plot_defaults()
    maxele = Maxele(os.path.join(KAT_EX_PATH, "maxele.63.nc"), crs="EPSG:4326")
    maxele.tricontourf(
        cbar=True, levels=20, label="Maxmimum Elevation [m]", vmin=0, vmax=3
    )
    plt.xlabel("Longitude [$^{\circ}$E]")
    plt.ylabel("Latitude [$^{\circ}$N]")

    plt.savefig(os.path.join(FIGURE_PATH, "example-whole-domain-maxele.png"))

    if in_notebook():
        plt.show()
    else:
        plt.clf()


def maxvel() -> None:
    """
    Max velocity.
    """
    plot_defaults()
    maxvel = Maxvel(os.path.join(KAT_EX_PATH, "maxvel.63.nc"), crs="EPSG:4326")
    maxvel.tricontourf(
        cbar=True, levels=20, label="Maxmimum Velocity [m s$^{-1}$]", vmin=0, vmax=3
    )
    plt.xlabel("Longitude [$^{\circ}$E]")
    plt.ylabel("Latitude [$^{\circ}$N]")

    plt.savefig(os.path.join(FIGURE_PATH, "example-whole-domain-maxvel.png"))

    if in_notebook():
        plt.show()
    else:
        plt.clf()


def multiplot(files=("fort.217", "fort.218")) -> None:
    """
    Multiplot.

    # U
    # V
    # Pressure
    # sharex, sharey

    Args:
        files (tuple, optional): Defaults to ("fort.217", "fort.218").
    """

    plot_defaults()
    p_da = plot_units(
        xr.open_dataarray(os.path.join(DATA_PATH, files[0] + ".nc")),
        x_dim="lon",
        y_dim="lat",
    )
    wsp_ds = plot_units(
        xr.open_dataset(os.path.join(DATA_PATH, files[1] + ".nc")),
        x_dim="lon",
        y_dim="lat",
    )

    fig, axs = plt.subplots(3, 1, sharey=True, sharex=True, figsize=get_dim(ratio=2))
    time = 40
    p_da.isel(time=time).plot(ax=axs[0], cmap=cmo.dense_r)
    axs[0].set_xlabel("")
    wsp_ds.isel(time=time).U10.plot(ax=axs[1], cmap=cmap("delta"))
    axs[1].set_title("")
    axs[1].set_xlabel("")
    wsp_ds.isel(time=time).V10.plot(ax=axs[2], cmap=cmap("delta"))
    axs[2].set_title("")
    label_subplots(axs, override="outside")

    plt.savefig(os.path.join(FIGURE_PATH, f"example_inputs{files[0]}{files[1]}.png"))

    if in_notebook():
        plt.show()
    else:
        plt.clf()


def multiplot_animate(
    # files=("fort.217", "fort.218"),
    files: str = ("fort.221", "fort.222"),
    video_path: str = os.path.join(FIGURE_PATH, "animation.mp4"),
) -> None:
    """
    Args:
        files (tuple, optional): Defaults to ("fort.221", "fort.222").
        video_path (str, optional): Defaults to os.path.join(FIGURE_PATH, "animation.mp4").
    """
    # U, V
    # Pressure
    # sharex, sharey
    matplotlib.use("Agg")

    plot_defaults()
    p_da = plot_units(xr.open_dataarray(os.path.join(DATA_PATH, files[0] + ".nc")))
    wsp_ds = plot_units(xr.open_dataset(os.path.join(DATA_PATH, files[1] + ".nc")))

    def gen_frame_func() -> Callable:
        """Create imageio frame function for `xarray.DataArray` visualisation.

        Returns:
            make_frame (Callable): function to create each frame.

        """
        vmin_p = p_da.min(skipna=True)
        vmax_p = p_da.max(skipna=True)
        vmin_v, vmax_v = lim(wsp_ds.U10.values, balance=True)

        def make_frame(index: int) -> np.ndarray:
            """Make an individual frame of the animation.

            Args:
                index (int): The time index.

            Returns:
                image (np.array): np.frombuffer output
                    that can be fed into imageio

            """
            fig, axs = plt.subplots(
                3, 1, sharey=True, sharex=True, figsize=get_dim(ratio=2)
            )
            p_da.isel(time=index).plot(
                ax=axs[0], vmin=vmin_p, vmax=vmax_p, cmap=cmo.dense_r
            )
            axs[0].set_xlabel("")
            wsp_ds.isel(time=index).U10.plot(
                ax=axs[1], vmin=vmin_v, vmax=vmax_v, cmap=cmap("delta")
            )
            axs[1].set_title("")
            axs[1].set_xlabel("")
            wsp_ds.isel(time=index).V10.plot(
                ax=axs[2], vmin=vmin_v, vmax=vmax_v, cmap=cmap("delta")
            )
            axs[2].set_title("")
            label_subplots(axs, override="outside")
            plt.tight_layout()
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()

            return image

        return make_frame

    def xarray_to_video(video_path: str, fps: int = 5) -> None:
        """Generate video of an `xarray.DataArray`.

        Args:
            video_path (str): output path to save.
            fps (int, optional): frames per second.

        """
        video_indices = list(range(p_da.sizes["time"]))
        make_frame = gen_frame_func()
        imageio.mimsave(
            video_path,
            [make_frame(index) for index in tqdm(video_indices, desc=video_path)],
            fps=fps,
        )
        print("Video " + video_path + " made.")

    xarray_to_video(video_path, fps=5)


if __name__ == "__main__":
    # python src/plot/adcirc.py
    maxele()
    maxvel()
    multiplot()
    multiplot(files=("fort.221", "fort.222"))
    multiplot(files=("fort.223", "fort.224"))
    # multiplot_animate()
