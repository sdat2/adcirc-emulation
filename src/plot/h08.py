"""Hurricane Holland Model 2008."""
import os
import numpy as np
import matplotlib.pyplot as plt
from sithom.plot import plot_defaults, label_subplots
from sithom.time import timeit
from sithom.misc import in_notebook
from src.constants import FIGURE_PATH
from src.conversions import knots_to_ms
from src.data_loading.ibtracs import kat_stats
from src.models.h08 import fit_h08


@timeit
def katrina_h08() -> None:
    """
    Katrina Hurricane 2008.
    """
    ds = kat_stats() # .isel(date_time=2)
    pc = ds["usa_pres"].values
    pn = ds["usa_poci"].values
    r64 = np.mean(ds["usa_r64"].values)
    vmax = ds["usa_wind"].values
    rmax = ds["usa_rmw"].values

    plot_defaults()
    radii = np.linspace(1, 1e6, num=int(1e4))

    _, axs = plt.subplots(2, 1, sharex=True)
    speeds = knots_to_ms([64, 50, 34])
    distances = [ds[x].values.mean() for x in ["usa_r64", "usa_r50", "usa_r34"]]
    axs[0].scatter(rmax, vmax)
    axs[0].scatter(distances, speeds)
    axs[0].set_ylabel("Azimuthal velocity (m s$^{-1}$)")
    axs[1].set_ylabel("Pressure (Pa)")
    axs[1].set_xlabel("Distance [m]")

    for density in [1, 1.1, 1.25, 2]:
        h08v_fit, h08p_fit, xn = fit_h08(
            float(rmax),
            float(vmax),
            float(pc),
            float(pn),
            float(r64),
            distances,
            speeds,
            density=density,
        )
        axs[0].plot(
            radii, h08v_fit(radii), label=f"density={density} kg m**-3, xn={xn!a}"
        )
        axs[1].plot(
            radii, h08p_fit(radii), label=f"density={density} kg m**-3, xn={xn!a}"
        )

    axs[1].legend()
    label_subplots(axs)

    plt.savefig(os.path.join(FIGURE_PATH, "Katrina_H08_DoublePanel.png"))

    if in_notebook():
        plt.show()
    else:
        plt.clf()


if __name__ == "__main__":
    # python src/plot/h08.py
    katrina_h08()
