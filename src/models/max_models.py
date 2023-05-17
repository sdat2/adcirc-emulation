import os
from typing import Tuple
import numpy as np
import xarray as xr
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import wandb
from sithom.plot import plot_defaults, lim, label_subplots
from sithom.place import BoundingBox
from sithom.time import timeit
from src.constants import NO_BBOX, NEW_ORLEANS, FIGURE_PATH, DATA_PATH, FEATURE_LIST, LABEL_LIST, SYMBOL_LIST, LABEL_DICT
from src.preprocessing.sel import trim_tri
# get data from weights and biases


@timeit
def generate_max_parray_and_output(
    version=0, bbox: BoundingBox = NO_BBOX
) -> Tuple[np.ndarray, np.ndarray, xr.Dataset]:
    # load artifact dataset
    # maybe we can add a loop here to get all the artifacts with different versions
    run = wandb.init()
    artifact = run.use_artifact(
        f"sdat2/6d_individual_version2/output_dataset:v{version}", type="dataset"
    )
    artifact_dir = artifact.download()
    cds_a = xr.open_dataset(os.path.join(artifact_dir, "combined_ds.nc"))

    # turn 7 parameters into array
    parray = cds_a[FEATURE_LIST].to_array().values
    # "vmax",
    # value order is: angle, speed, point_east, rmax, pc, vmax, xn
    ### output array
    oa = cds_a[["zeta_max"]]
    indices = bbox.indices_inside(cds_a["lon"].values, cds_a["lat"].values)
    oa = oa.isel(node=indices)
    output_array = oa.to_array().values.transpose()

    return parray.reshape(1, -1), output_array.reshape(1, -1), cds_a


@timeit
def make_all_plots(regenerate=True) -> None:
    bbox = NO_BBOX.pad(2)
    bbox_plot = NO_BBOX
    bbox_plot.lat[0] = NO_BBOX.lat[0] - 1
    bbox_plot.lat[1] = NO_BBOX.lat[1]
    data_path = os.path.join(DATA_PATH, "max_sensitivities")
    figure_path = os.path.join(FIGURE_PATH, "max_sensitivities")
    num = 400
    os.makedirs(figure_path, exist_ok=True)
    os.makedirs(data_path, exist_ok=True)
    plot_defaults()

    def load_data_from_scratch() -> Tuple[np.ndarray, np.ndarray, xr.Dataset]:
        parray_list = []
        output_array_list = []

        for i in range(num):
            print(i)
            parray, output_array, cds_a = generate_max_parray_and_output(
                version=i, bbox=bbox
            )
            parray_list.append(parray)
            output_array_list.append(output_array)
            print(parray.shape, output_array.shape)

        parray_mult = np.concatenate(parray_list, axis=0)

        oa_mult = np.concatenate(
            [output_array_list[i].reshape(1, -1) for i in range(len(output_array_list))]
        )

        return parray_mult, oa_mult, cds_a

    if os.path.exists(os.path.join(data_path, "parray_mult.npy")) and not regenerate:
        parray_mult = np.load(os.path.join(data_path, "parray_mult.npy"))
        oa_mult = np.load(os.path.join(data_path, "oa_mult.npy"))
        cds_a = xr.open_dataset(os.path.join(data_path, "cds_a.nc"))
    else:
        parray_mult, oa_mult, cds_a = load_data_from_scratch()
        # parray_mult.save(os.path.join(data_path, "parray_mult.npy"))
        # oa_mult.save(os.path.join(data_path, "oa_mult.npy"))
        np.save(os.path.join(data_path, "parray_mult.npy"), parray_mult)
        np.save(os.path.join(data_path, "oa_mult.npy"), oa_mult)
        cds_a.to_netcdf(os.path.join(data_path, "cds_a.nc"))

    # Importance plots

    # print("oa_mult.shape", oa_mult.shape)

    def return_importances(index: int = 27) -> np.ndarray:
        # TODO: Replace with GP version!!!
        # print(index)
        model = DecisionTreeRegressor()
        model.fit(parray_mult, oa_mult[:, index])
        importances = model.feature_importances_
        return importances

    importance_array = np.array(
        [return_importances(i).tolist() for i in range(oa_mult.shape[1])]
    )
    max_importance = np.argmax(importance_array, axis=1)

    for i in range(len(FEATURE_LIST)):
        lon, lat, triangles = trim_tri(
            cds_a.lon.values, cds_a.lat.values, cds_a.triangle.values - 1, bbox
        )
        vmin, vmax = lim(importance_array[:, i], percentile=0, balance=True)
        vmin, vmax = np.min([-vmax, vmin]), np.max([-vmin, vmax])
        levels = np.linspace(vmin, vmax, num=400)
        cbar_levels = np.linspace(vmin, vmax, num=7)

        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent(bbox_plot.cartopy(), crs=ccrs.PlateCarree())
        # add a green-yellow backgroud here
        ax.set_facecolor("#d1ffbd")
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
        plt.tricontourf(
            lon,
            lat,
            triangles,
            importance_array[:, i],
            vmin=vmin,
            vmax=vmax,
            levels=levels,
            # cmap="cmo.balance",
        )

        ax = plt.gca()
        cbar = plt.colorbar(
            label="Importance of "
            + LABEL_LIST[i]  # FEATURE_LIST[i].replace("_", " ").capitalize()
        )
        cbar.set_ticks(cbar_levels)
        cbar.set_ticklabels(["{:.2f}".format(x) for x in cbar_levels.tolist()])
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
        plt.savefig(os.path.join(figure_path, "importance_" + FEATURE_LIST[i] + ".png"))
        plt.clf()

    # plot all importances in one plot

    # plot most important feature

    lon, lat, triangles = trim_tri(
        cds_a.lon.values, cds_a.lat.values, cds_a.triangle.values - 1, bbox
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

    plt.scatter(lon, lat, c=max_importance, cmap=cmap, vmin=vmin, vmax=vmax, s=15)

    ax = plt.gca()
    cbar = plt.colorbar(label="Most important feature")
    cbar.set_ticks(cbar_levels)
    cbar.set_ticklabels([SYMBOL_LIST[int(x)] for x in cbar_levels])
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
    plt.savefig(os.path.join(figure_path, "max_importance.png"))
    plt.clf()

    def return_correlation(index: int = 27) -> np.ndarray:
        return np.array(
            [
                np.corrcoef(parray_mult[:, i], oa_mult[:, index])[0, 1]
                for i in range(parray_mult.shape[1])
            ]
        )

    # correlation plots

    correlation_array = np.array(
        [return_correlation(i).tolist() for i in range(oa_mult.shape[1])]
    )
    lon, lat, triangles = trim_tri(
        cds_a.lon.values, cds_a.lat.values, cds_a.triangle.values - 1, bbox
    )

    for i in range(len(FEATURE_LIST)):
        vmin, vmax = lim(correlation_array[:, i], percentile=0, balance=True)
        vmin, vmax = np.min([-vmax, vmin]), np.max([-vmin, vmax])
        levels = np.linspace(vmin, vmax, num=400)
        cbar_levels = np.linspace(vmin, vmax, num=7)

        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent(bbox_plot.cartopy(), crs=ccrs.PlateCarree())
        # add a green-yellow backgroud here
        ax.set_facecolor("#d1ffbd")
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
        plt.tricontourf(
            lon,
            lat,
            triangles,
            correlation_array[:, i],
            vmin=vmin,
            vmax=vmax,
            levels=levels,
            cmap="cmo.balance",
        )

        ax = plt.gca()
        cbar = plt.colorbar(
            label="Correlation " + FEATURE_LIST[i].replace("_", " ").capitalize()
        )
        cbar.set_ticks(cbar_levels)
        cbar.set_ticklabels(["{:.2f}".format(x) for x in cbar_levels.tolist()])
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
        plt.savefig(
            os.path.join(figure_path, "correlation_" + FEATURE_LIST[i] + ".png")
        )
        plt.clf()

    # correlation plots all together
    vmin, vmax = lim(correlation_array, percentile=0, balance=True)
    vmin, vmax = np.min([-vmax, vmin]), np.max([-vmin, vmax])
    levels = np.linspace(vmin, vmax, num=400)
    cbar_levels = np.linspace(vmin, vmax, num=7)

    fig, axs = plt.subplots(
        3, 2, figsize=(6, 6), subplot_kw={"projection": ccrs.PlateCarree()}
    )

    for i in range(len(FEATURE_LIST)):
        ax = axs.transpose().ravel()[i]
        ax.set_extent(bbox_plot.cartopy(), crs=ccrs.PlateCarree())
        # add a green-yellow backgroud here
        ax.set_facecolor("#d1ffbd")
        ax.add_feature(cartopy.feature.LAKES, alpha=0.5, color="lightblue")
        ax.add_feature(cartopy.feature.RIVERS, alpha=0.5, color="lightblue")
        im = ax.tricontourf(
            lon,
            lat,
            triangles,
            correlation_array[:, i],
            vmin=vmin,
            vmax=vmax,
            levels=levels,
            cmap="cmo.balance",
        )
        ax.set_title(LABEL_LIST[i])
        ax.plot(
            NEW_ORLEANS.lon, NEW_ORLEANS.lat, marker=".", markersize=4, color="purple"
        )
        ax.text(
            NEW_ORLEANS.lon - 0.35,
            NEW_ORLEANS.lat - 0.16,
            "New Orleans",
            fontsize=6,
            color="purple",
        )
        ax.set_yticks(
            [
                x
                for x in range(
                    int((bbox_plot.lat[0] // 1) + 1), int((bbox_plot.lat[1] // 1) + 1)
                )
            ],
            crs=ccrs.PlateCarree(),
        )
        ax.set_xticks(
            [
                x
                for x in range(
                    int((bbox_plot.lon[0] // 1) + 1), int((bbox_plot.lon[1] // 1) + 1)
                )
            ],
            crs=ccrs.PlateCarree(),
        )
        ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
        ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)

    label_subplots(axs.transpose(), override="outside", fontsize=12)
    fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar_ax = fig.add_axes([1.02, 0.15, 0.05, 0.7])

    cbar = fig.colorbar(
        im,
        cax=cbar_ax,
        label="Correlation with maximum sea surface height, $\eta_{\mathrm{max}}$, [dimensionless]",
    )
    cbar_levels = np.linspace(vmin, vmax, num=7)
    cbar.set_ticks(cbar_levels)
    cbar.set_ticklabels(["{:.2f}".format(x) for x in cbar_levels.tolist()])
    plt.savefig(os.path.join(figure_path, "correlation_all.png"), bbox_inches="tight")
    plt.clf()

    # importance plots all together
    vmin, vmax = lim(importance_array, percentile=0, balance=False)
    # vmin, vmax = np.min([-vmax, vmin]), np.max([-vmin, vmax])
    levels = np.linspace(vmin, vmax, num=400)
    cbar_levels = np.linspace(vmin, vmax, num=7)

    fig, axs = plt.subplots(
        3, 2, figsize=(6, 6), subplot_kw={"projection": ccrs.PlateCarree()}
    )

    for i in range(len(FEATURE_LIST)):
        ax = axs.transpose().ravel()[i]
        ax.set_extent(bbox_plot.cartopy(), crs=ccrs.PlateCarree())
        # add a green-yellow backgroud here
        ax.set_facecolor("#d1ffbd")
        ax.add_feature(cartopy.feature.LAKES, alpha=0.5, color="lightblue")
        ax.add_feature(cartopy.feature.RIVERS, alpha=0.5, color="lightblue")
        im = ax.tricontourf(
            lon,
            lat,
            triangles,
            importance_array[:, i],
            vmin=vmin,
            vmax=vmax,
            levels=levels,
            cmap="cmo.amp",
        )
        ax.set_title(LABEL_LIST[i])
        ax.plot(
            NEW_ORLEANS.lon, NEW_ORLEANS.lat, marker=".", markersize=4, color="purple"
        )
        ax.text(
            NEW_ORLEANS.lon - 0.35,
            NEW_ORLEANS.lat - 0.16,
            "New Orleans",
            fontsize=6,
            color="purple",
        )
        ax.set_yticks(
            [
                x
                for x in range(
                    int((bbox_plot.lat[0] // 1) + 1), int((bbox_plot.lat[1] // 1) + 1)
                )
            ],
            crs=ccrs.PlateCarree(),
        )
        ax.set_xticks(
            [
                x
                for x in range(
                    int((bbox_plot.lon[0] // 1) + 1), int((bbox_plot.lon[1] // 1) + 1)
                )
            ],
            crs=ccrs.PlateCarree(),
        )
        ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
        ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)

    label_subplots(axs.transpose(), override="outside", fontsize=12)
    fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar_ax = fig.add_axes([1.02, 0.15, 0.05, 0.7])

    cbar = fig.colorbar(
        im,
        cax=cbar_ax,
        label="Importance of feature for predicting, $\eta_{\mathrm{max}}$, [dimensionless]",
    )
    cbar_levels = np.linspace(vmin, vmax, num=7)
    cbar.set_ticks(cbar_levels)
    cbar.set_ticklabels(["{:.2f}".format(x) for x in cbar_levels.tolist()])
    plt.savefig(os.path.join(figure_path, "importance_all.png"), bbox_inches="tight")
    plt.clf()


if __name__ == "__main__":
    # python src/models/max_models.py
    make_all_plots(regenerate=False)
