"""Tidal Comparison Plots."""
import os
import numpy as np
import xarray as xr
import datetime
import netCDF4 as nc
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import cartopy
    import cartopy.crs as ccrs
except ImportError:
    print("cartopy not installed")
from sithom.plot import plot_defaults, label_subplots, get_dim
from src.data_loading.tides import filtered_tidal_gauges
from src.data_loading.adcirc import select_coastal_cells
from src.constants import KAT_EX_PATH, FIGURE_PATH, NO_BBOX
from src.plot.map import add_features


def tide_plot(stationid: int = 0) -> None:
    """
    Tide plot.

    Args:
        stationid (int, optional): stationid. Defaults to 0.
    """
    # python src/plot/tides.py
    tds = filtered_tidal_gauges()
    print(tds)
    psc = tds.isel(stationid=stationid)
    print((float(psc.lon.values), float(psc.lat.values)))

    ds = select_coastal_cells(float(psc.lon.values), float(psc.lat.values), number=10)

    plot_defaults()
    fig, axs = plt.subplots(2, 1, figsize=get_dim(ratio=1))
    axs[1].remove()
    axs[1] = fig.add_subplot(2, 1, 2, projection=ccrs.PlateCarree())
    psc.water_level.plot(ax=axs[0])
    ds.Height.plot.line(ax=axs[0], hue="point", alpha=0.5)
    axs[0].set_title(str(psc.name.values))
    sns.move_legend(axs[0], loc="upper left", bbox_to_anchor=(1, 1.05))
    add_features(axs[1])
    tri_plot_filter(axs[1])

    axs[1].scatter(psc.lon.values, psc.lat.values, s=5, marker="+", alpha=0.8)

    for point in ds.point.values:
        axs[1].scatter(
            ds.sel(point=point).lon.values,
            ds.sel(point=point).lat.values,
            s=3,
            label=str(point),
            alpha=0.8,
        )

    axs[1].set_xlabel("Longitude [$^{\circ}$E]")
    axs[1].set_ylabel("Latitude [$^{\circ}$N]")

    plt.legend()
    sns.move_legend(axs[1], loc="upper left", bbox_to_anchor=(1, 1.05))
    label_subplots(axs)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_PATH, "tide_gauge" + str(stationid) + ".png"))
    plt.clf()


def depth_plot() -> None:
    """
    Depth plot.
    """
    f63 = nc.Dataset(os.path.join(KAT_EX_PATH, "fort.63.nc"))
    x = f63["x"][:].data.ravel()
    y = f63["y"][:].data.ravel()
    tri = (f63["element"][:] - 1).data
    depth = f63["depth"][:].data
    print("tri", tri.shape, type(tri))
    print("x", x.shape, type(x))
    print("y", y.shape, type(y))

    plt.tricontourf(x, y, tri, depth)
    plt.colorbar(label="Depth [m]")


def tri_plot() -> None:
    f63 = nc.Dataset(os.path.join(KAT_EX_PATH, "fort.63.nc"))
    x = f63["x"][:].data.ravel()
    y = f63["y"][:].data.ravel()
    tri = (f63["element"][:] - 1).data
    print("tri", tri.shape, type(tri))
    print("x", x.shape, type(x))
    print("y", y.shape, type(y))
    plt.triplot(x, y, tri, linewidth=0.5)


def tri_plot_filter(ax: matplotlib.axes.Axes) -> None:
    f63 = nc.Dataset(os.path.join(KAT_EX_PATH, "fort.63.nc"))
    x = f63["x"][:].data.ravel()
    y = f63["y"][:].data.ravel()
    tri = (f63["element"][:] - 1).data

    @np.vectorize
    def in_new_orleans(xi, yi):
        return (
            xi > NO_BBOX.lon[0]
            and xi < NO_BBOX.lon[1]
            and yi > NO_BBOX.lat[0]
            and yi < NO_BBOX.lat[1]
        )

    tindices = in_new_orleans(x, y)
    indices = np.where(tindices)[0]
    new_indices = np.where(indices)[0]
    neg_indices = np.where(~tindices)[0]
    tri_list = tri.tolist()
    new_tri_list = []
    for el in tri_list:
        if np.any([x in neg_indices for x in el]):
            continue
        else:
            new_tri_list.append(el)

    tri_new = np.array(new_tri_list)
    print(tri.shape)
    print(tri_new.shape)
    print(indices[:10])
    print(new_indices[:10])
    tri_new = np.select(
        [tri_new == x for x in indices.tolist()], new_indices.tolist(), tri_new
    )
    print(tri_new.max())
    print(tri_new.min())

    ax.triplot(x[tindices], y[tindices], tri_new, linewidth=0.5, color="grey")


if __name__ == "__main__":
    # python src/plot/tides.py
    tds = filtered_tidal_gauges()
    stations = len(tds["stationid"].values)
    _ = [tide_plot(x) for x in range(stations)]
