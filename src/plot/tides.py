"""Tidal Comparison Plots."""
import os
import xarray as xr
import datetime
import netCDF4 as nc
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
from src.constants import KAT_EX_PATH, FIGURE_PATH
from src.plot.map import add_features


def tide_plot(stationid=0):
    # python src/plot/tides.py
    tds = filtered_tidal_gauges()
    print(tds)
    psc = tds.isel(stationid=stationid)
    print((float(psc.lon.values), float(psc.lat.values)))

    lons, lats, heights = select_coastal_cells(
        float(psc.lon.values), float(psc.lat.values)
    )
    start = datetime.datetime(year=2005, month=8, day=19, hour=5)
    time_step = datetime.timedelta(hours=1, minutes=20)
    # start = datetime.datetime(year=2005, month=8, day=21, hour=18)
    # time_step = datetime.timedelta(hours=1)
    # time_step = datetime.timedelta(hours=1, minutes=15)
    # start = datetime.datetime(year=2005, month=8, day=20, hour=18)

    print(heights.shape)
    ds = xr.Dataset(
        data_vars=dict(Height=(["time", "point"], heights)),
        coords=dict(
            lon=(["point"], lons),
            lat=(["point"], lats),
            time=[start + i * time_step for i in range(heights.shape[0])],
        ),
    )
    ds["Height"].attrs["units"] = "m"
    print(ds)

    plot_defaults()
    fig, axs = plt.subplots(2, 1, figsize=get_dim(ratio=1))
    axs[1].remove()
    axs[1] = fig.add_subplot(2, 1, 2, projection=ccrs.PlateCarree())
    psc.water_level.plot(ax=axs[0])
    ds.Height.plot.line(ax=axs[0], hue="point", alpha=0.5)
    axs[0].set_title(str(psc.name.values))
    sns.move_legend(axs[0], loc="upper left", bbox_to_anchor=(1, 1.05))

    axs[1].scatter(psc.lon.values, psc.lat.values, s=4)
    for point in ds.point.values:
        axs[1].scatter(
            ds.sel(point=point).lon.values,
            ds.sel(point=point).lat.values,
            s=2,
            label=str(point),
        )
    add_features(axs[1])

    axs[1].set_xlabel("Longitude [$^{\circ}$E]")
    axs[1].set_ylabel("Latitude [$^{\circ}$N]")

    plt.legend()
    sns.move_legend(axs[1], loc="upper left", bbox_to_anchor=(1, 1.05))

    label_subplots(axs)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_PATH, "tide_gauge" + str(stationid) + ".png"))
    plt.clf()


if __name__ == "__main__":
    # python src/plot/tides.py
    # tds = filtered_tidal_gauges()
    # stations = len(tds["stationid"].values)
    # _ = [tide_plot(x) for x in range(stations)]
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
    # import matplotlib

    # matplotlib.tri.Triangulation(
    #    f63["x"][:], f63["y"][:], triangles=(f63["element"][:] - 1)
    # )

    plt.show()
