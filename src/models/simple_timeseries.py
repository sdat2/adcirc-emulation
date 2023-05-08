"""
Simple timeseries models.
"""
import os
from typing import Tuple
import numpy as np
import xarray as xr
import wandb
import omegaconf
from omegaconf import OmegaConf, DictConfig
import hydra
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sithom.time import timeit
from sithom.place import BoundingBox
from src.constants import CONFIG_PATH, DATA_PATH, NO_BBOX

# FEATURE_LIST = [#
FEATURE_LIST = ["angle", "speed", "point_east", "rmax", "pc", "xn"]


def make_8d_data(num=400, bbox: BoundingBox = NO_BBOX) -> None:
    """
    Make 8d data netcdf for reading by the script.

    Downloads data from weights and biases, and then formats it
    for machine learning.

    Args:
        num (int, optional): numb of versions to download. Defaults to 400.
        bbox (BoundingBox, optional): bbox. Defaults to NO_BBOX.
    """
    run = wandb.init()

    def generate_parray2d_and_output(
        version=0,
    ) -> Tuple[np.ndarray, np.ndarray, xr.Dataset]:
        """
        Generate parray2d and output.

        Args:
            version (int, optional): version of artifact. Defaults to 0.

        Returns:
            Tuple[np.ndarray, np.ndarray, xr.Dataset]: parray2d, output, and dataset.
        """
        # tide feature needed for time varying version to work.
        artifact = run.use_artifact(
            f"sdat2/6d_individual_version2/output_dataset:v{version}", type="dataset"
        )
        artifact_dir = artifact.download()
        cds_a = xr.open_dataset(os.path.join(artifact_dir, "combined_ds.nc"))

        # turn 6 parameters into array
        parray = cds_a[FEATURE_LIST].to_array().values

        # get clat and clon (centers of cyclone over time)
        clat = cds_a["clat"].values
        clon = cds_a["clon"].values

        # so we want to format the inputs for machine learning, so we need to get the input data into a 2D array
        # parray needs to become as long as clat and clon (repeated entries for 56 timesteps)
        # clat and clon need to appended to this array --> a total of 7 + 2 = 9 parameters * 56 timesteps = 504 entries
        parray_2d = np.array([parray for _ in range(len(clat))])
        # now we need to append clat and clon to this array
        parray_2d = np.append(parray_2d, clat.reshape(-1, 1), axis=1)
        parray_2d = np.append(parray_2d, clon.reshape(-1, 1), axis=1)
        # parray_2d = np.append(parray_2d, clat, axis=1)
        # value order is: angle, speed, point_east, rmax, pc, vmax, xn, clat, clon

        ### output array
        oa = cds_a[["zeta", "u-vel", "v-vel"]].interp(
            {"output_time": cds_a["input_time"]}
        )

        (indices,) = NO_BBOX.indices_inside(cds_a["lon"].values, cds_a["lat"].values)
        oa = oa.isel(node=indices)

        output_array = oa.to_array().values.transpose()

        return parray_2d, output_array, cds_a

    parray_2d_list, output_array_list = [], []
    for version in range(num):
        parray_2d, output_array, cds_a = generate_parray2d_and_output(version=version)
        parray_2d_list.append(parray_2d)
        output_array_list.append(output_array)

    output_array_4d = np.array(output_array_list)
    parray_2d_3d = np.array(parray_2d_list)
    indices = NO_BBOX.indices_inside(cds_a["lon"].values, cds_a["lat"].values)

    ds8 = xr.Dataset(
        data_vars={
            "x": (["exp", "time", "param"], parray_2d_3d),
            "y": (["exp", "node", "time", "output"], output_array_4d),
        },
        coords={
            "exp": range(output_array_4d.shape[0]),
            "node": range(1, output_array_4d.shape[1] + 1),
            "time": cds_a["input_time"].values,
            "param": FEATURE_LIST + ["clon", "clat"],
            "output": ["zeta", "u-vel", "v-vel"],
            "lat": (["node"], cds_a.isel(node=indices[0]).lat.values),
            "lon": (["node"], cds_a.isel(node=indices[0]).lon.values),
        },
    )

    ds8.to_netcdf(os.path.join(DATA_PATH, "ds8.nc"))


@timeit
def load_8d_data() -> xr.Dataset:
    """
    Load the 8d dataset.

    Returns:
        xr.Dataset: xarray dataset.
    """
    ds8 = xr.open_dataset(os.path.join(DATA_PATH, "ds8.nc"))
    return ds8


@timeit
def rescale_ds(ds8: xr.Dataset) -> xr.Dataset:
    """
    Rescale the dataset using config array for standard variables and mean and std for the rest.

    Args:
        ds8 (xr.Dataset): unscaled dataset.

    Returns:
        xr.Dataset: scaled dataset.
    """
    # takes around 7 seconds.
    # rescale
    cfg = OmegaConf.load(os.path.join(CONFIG_PATH, "sixd.yaml"))
    # print(cfg)
    mu_list = []
    sigma_list = []
    ds8r = ds8.copy(deep=True)

    # normalizing the variables chosen by latin hypercube search.
    for i, param in enumerate(ds8.param.values[:-2]):
        mu = (cfg[param].max + cfg[param].min) / 2
        sigma = (cfg[param].max - cfg[param].min) / 2
        ds8r.x[:, :, i] = (ds8.x[:, :, i] - mu) / sigma
        mu_list.append(mu)
        sigma_list.append(sigma)

    for i, param in enumerate(ds8.param.values[-2:]):
        mu = np.nanmean(ds8.x[:, :, i + 6].values)
        sigma = np.nanstd(ds8.x[:, :, i + 6].values)
        if sigma == 0.0:
            sigma = 1
        # assigment
        ds8r.x[:, :, i + 6] = (ds8.x[:, :, i + 6] - mu) / sigma
        mu_list.append(mu)
        sigma_list.append(sigma)

    rescale_x_ds = xr.Dataset(
        data_vars={"mu_x": (["param"], mu_list), "sigma_x": (["param"], sigma_list)}
    )

    ds8r = xr.merge([ds8r, rescale_x_ds])

    mu_list = []
    sigma_list = []

    for i, output in enumerate(ds8.output.values):
        mu_list.append([])
        sigma_list.append([])
        for j, node in enumerate(ds8.node.values):
            mu = np.nanmean(ds8.y[:, j, :, i].values)
            sigma = np.nanstd(ds8.y[:, j, :, i].values)
            if sigma == 0.0:
                sigma = 0.0001
            ds8r.y[:, j, :, i] = (ds8.y[:, j, :, i] - mu) / sigma
            mu_list[-1].append(mu)
            sigma_list[-1].append(sigma)

    rescale_y_ds = xr.Dataset(
        data_vars={
            "mu_y": (["output", "node"], np.array(mu_list)),
            "sigma_y": (["output", "node"], np.array(sigma_list)),
        }
    )

    ds8r = xr.merge([ds8r, rescale_y_ds])

    return ds8r


@timeit
def get_simple_split(
    ds8r: xr.Dataset, index=26
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simple train test split using sklearn.

    Args:
        ds8r (xr.Dataset): rescaled dataset.
        index (int, optional): which node. Defaults to 26.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: x_train, x_test, y_train, y_test
    """
    # ds8r

    ex8d = ds8r.isel(node=index, output=0)
    # simplest possible split
    x_values = ex8d.x.values
    y_values = ex8d.y.values
    # print(x_values.shape, y_values.shape)
    # we want to ravel first two dimensions
    x_values = x_values.reshape(-1, x_values.shape[-1])
    y_values = np.nan_to_num(y_values.reshape(-1, y_values.shape[-1]).ravel())
    # print(x_values.shape, y_values.shape)
    # let's just predict zeta for now
    y_values = y_values  # [  # .reshape(-1, 1)
    # print(x_values.shape, y_values.shape)
    x_train, x_test, y_train, y_test = train_test_split(
        x_values, y_values, random_state=1
    )
    # print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    return x_train, x_test, y_train, y_test


@timeit
def get_exp_split(
    ds8r: xr.Dataset, index=26, split_index=200
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split single index by exp instead of randomly.
    This allows us to have random split out instead
    of full thing.

    Args:
        ds8r (xr.Dataset): rescaled dataset.
        index (int, optional): Defaults to 26.
        split_index (int, optional): Defaults to 220.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: x_train, x_test, y_train, y_test
    """
    ex8d = ds8r.isel(node=index, exp=slice(0, split_index), output=0)
    x_values = ex8d.x.values
    y_values = ex8d.y.values
    x_train = x_values.reshape(-1, x_values.shape[-1])
    y_train = np.nan_to_num(y_values.reshape(-1, y_values.shape[-1]).ravel())
    ex8d = ds8r.isel(node=index, exp=slice(split_index, 286), output=0)
    x_values = ex8d.x.values
    y_values = ex8d.y.values
    x_test = x_values.reshape(-1, x_values.shape[-1])
    y_test = np.nan_to_num(y_values.reshape(-1, y_values.shape[-1]).ravel())

    return x_train, x_test, y_train, y_test


def train_mlp(x_train: np.ndarray, y_test: np.ndarray) -> any:
    """
    Train a multi-layer perceptron with x_train, y_train.

    Args:
        x_train (np.ndarray): x training data.
        y_test (np.ndarray): y training data.

    Returns:
        any: Trained multi layer perceptron.
    """
    raise NotImplementedError


if "__main__" == __name__:
    # python src/models/simple_timeseries.py
    ds8r = rescale_ds(load_8d_data())
    a = get_simple_split(ds8r)
    for i in a:
        print(i.shape)

    # go through all of the possible nodes.
    for i in range(len(ds8r.node.values)):
        a = get_exp_split(ds8r, index=i)
        for j in a:
            print(j.shape)

    # Get Experiment Split..
    a = get_exp_split(ds8r)
    for i in a:
        # x_train, x_test, y_train, y_test
        print(i.shape)

    print(len(ds8r.node.values))
