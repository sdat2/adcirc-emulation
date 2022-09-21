"""Plot fitting Holland 2010."""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sithom.misc import in_notebook
from sithom.plot import label_subplots, axis_formatter, plot_defaults
from src.constants import FIGURE_PATH
from src.conversions import knots_to_ms
from src.data_loading.ibtracs import (
    katrina,
    holland2010,
    holland_fitter_usa,
    landings_only,
)


def plot_example() -> None:
    """
    Plot example Holland Hurricane.
    """
    radii = np.linspace(0, 5.0e5, int(1e3))
    # velocities = holland2010(radii, bs_coeff=0.003, x_coeff=0.5, rmax=5000, vmax=50)
    ds = landings_only(katrina()).isel(date_time=2)

    velocities = holland2010(radii, 0.5, 8, ds["usa_rmw"].values, ds["usa_wind"].values)
    plt.plot(radii, velocities)
    plt.xlabel("Radius (m)")
    plt.ylabel("Velocity (m s$^{-1}$)")
    plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter("%.0e"))

    radii_kat = np.array(
        [np.mean(ds[var].values) for var in ["usa_r34", "usa_r50", "usa_r64"]]
    )
    speeds_kat = knots_to_ms(np.array([34, 50, 64]))

    plt.scatter(ds["usa_rmw"].values, ds["usa_wind"].values)
    plt.scatter(radii_kat, speeds_kat, c="red")

    # popt, func = holland_fitter_usa(ds)
    # print(popt)
    # plt.plot(radii, func(radii))

    plt.savefig(os.path.join(FIGURE_PATH, "holland_example.png"))
    if in_notebook():
        plt.show()
    else:
        plt.clf()


if __name__ == "__main__":
    plot_defaults()
    # python src/plot/holland.py
    plot_example()
    print([var for var in landings_only(katrina()).isel(date_time=2)])
