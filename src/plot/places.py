"""
Places.py.
"""
import os
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from sithom.plot import plot_defaults
from sithom.place import BoundingBox
from sithom.time import timeit
from src.constants import (
    FEATURE_LIST,
    SYMBOL_LIST,
    NEW_ORLEANS,
    FIGURE_PATH,
    NO_BBOX,
    DATA_PATH,
    LABEL_LIST,
    PLACES_D,
)
from src.preprocessing.sel import trim_tri


def plot_places(figure_path: str) -> None:
    bbox = NO_BBOX.pad(2)
    bbox_plot = NO_BBOX
    bbox_plot.lat[0] = NO_BBOX.lat[0] - 1
    bbox_plot.lat[1] = NO_BBOX.lat[1]
    plot_defaults()
    cds_a = xr.open_dataset(os.path.join(DATA_PATH, "max_sensitivities", "cds_a.nc"))

    lon, lat, triangles = trim_tri(
        cds_a.lon.values, cds_a.lat.values, cds_a.triangle.values - 1, bbox
    )
    print(cds_a)
    print([x for x in cds_a])
    lonb, latb, trib = trim_tri(
        cds_a.lon.values,
        cds_a.lat.values,
        cds_a.triangle.values - 1,
        bbox_plot,
        # cds_a.node.values,
    )
    vmin = -0.5
    vmax = len(FEATURE_LIST) - 0.5
    # 0, 1, 2, 3, 4, 5,

    r = len(FEATURE_LIST) - 1
    levels = [vmin + 0.5 + i for i in range(r)]  # np.linspace(vmin, vmax, num=r)
    cmap = cm.get_cmap("Set1", r + 1)
    cbar_levels = [vmin + 0.5 + i for i in range(r + 1)]

    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(bbox_plot.cartopy(), crs=ccrs.PlateCarree())
    # add a green-yellow backgroud here
    # ax.set_facecolor("#d1ffbd")
    ax.add_feature(cartopy.feature.LAKES, alpha=0.5, color="lightblue")
    # Why are the rivers not plotting in notebook?
    ax.add_feature(cartopy.feature.RIVERS, alpha=0.5, color="lightblue")
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
    plt.triplot(lon, lat, triangles, color="black")
    # plt.scatter(lon, lat, c=max_importance, cmap=cmap, vmin=vmin, vmax=vmax, s=15)
    # colors = ["b", "r", "g", "y", "k", "m", "c"][::-1]
    colors = ["b", "m", "c", "g", "y", "r", "k"]
    for i, key in enumerate(PLACES_D):
        plt.scatter(
            lonb[PLACES_D[key]],
            latb[PLACES_D[key]],
            label=key.replace("_", " ").capitalize(),
            color=colors[i],
        )

    ax = plt.gca()
    # cbar = plt.colorbar(label="Most important feature")
    # cbar.set_ticks(cbar_levels)
    # cbar.set_ticklabels([SYMBOL_LIST[int(x)] for x in cbar_levels])

    plt.xlabel("")
    plt.ylabel("")
    ax.set_yticks(
        [
            x
            for x in range(
                int((bbox_plot.lat[0] // 1) + 1),
                int((bbox_plot.lat[1] // 1) + 1),
            )
        ],
        crs=ccrs.PlateCarree(),
    )
    ax.set_xticks(
        [
            x
            for x in range(
                int((bbox_plot.lon[0] // 1) + 1),
                int((bbox_plot.lon[1] // 1) + 1),
            )
        ],
        crs=ccrs.PlateCarree(),
    )

    ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
    ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)
    plt.legend()
    plt.savefig(os.path.join(figure_path, "places.png"))
    plt.clf()


if __name__ == "__main__":
    # python src/plot/places.py
    plot_places(FIGURE_PATH)
