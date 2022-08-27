"""Panel plot."""
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from sithom.misc import in_notebook
from sithom.plot import label_subplots, set_dim, plot_defaults
from sithom.xr import plot_units, mon_increase
from sithom.time import timeit
from src.constants import KATRINA_ERA5_NC, FIGURE_PATH
from src.preprocessing.sel import mid_katrina


def panel_plot(input_ds: xr.Dataset, panel_array: np.ndarray) -> None:
    """
    Panel plotter for multiple fields. Can be generalised more. Removing ravelling could help.

    TODO: Add (automatic?) colormaps.

    Args:
        input_ds (xr.Dataset): Input dataset (single time slice)
        panel_array (np.ndarray): Input variable names to plot.

    """
    fig, axs = plt.subplots(
        panel_array.shape[0], panel_array.shape[1], sharex=True, sharey=True
    )

    label_subplots(axs, override="outside")
    num = len(axs.ravel())
    for i in range(num):
        var = panel_array.ravel()[i]
        ax = axs.ravel()[i]
        if var is not None:
            input_ds[panel_array.ravel()[i]].plot(ax=ax)
            ax.set_title("")
            if i % 2:
                ax.set_ylabel("")
            if i < num - 2:
                ax.set_xlabel("")
        else:
            ax.remove()

    set_dim(fig, fraction_of_line_width=2.4)


@timeit
def katrina_air() -> None:
    """Katrina ECMWF AIR variables panel plot mid Katrina."""
    plot_defaults()
    ds = xr.open_dataset(KATRINA_ERA5_NC)
    kat_ds = mid_katrina(mon_increase(plot_units(ds)))
    panel_array = np.array(
        [["u10", "v10"], ["d2m", "t2m"], ["msl", "sp"], ["tp", None]]
    )
    panel_plot(kat_ds, panel_array)
    plt.savefig(os.path.join(FIGURE_PATH, "mid_katrina_ecmwf_air_fields.png"))
    if not in_notebook():
        plt.clf()


if __name__ == "__main__":
    # python src/plot/panel.py
    katrina_air()
