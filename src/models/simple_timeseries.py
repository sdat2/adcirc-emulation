"""
Simple timeseries models.
"""
import os
from typing import Tuple
import numpy as np
import xarray as xr
import omegaconf
from omegaconf import OmegaConf, DictConfig
import hydra
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sithom.time import timeit
from src.constants import CONFIG_PATH, DATA_PATH


@timeit
def load_8d_data():
    ds8 = xr.open_dataset(os.path.join(DATA_PATH, "ds8.nc"))
    return ds8


@timeit
def rescale_ds(ds8: xr.Dataset) -> xr.Dataset:
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
    Get exp split.

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


if "__main__" == __name__:
    # python src/models/simple_timeseries.py
    ds8r = rescale_ds(load_8d_data())
    a = get_simple_split(ds8r)
    for i in a:
        print(i.shape)

    for i in range(len(ds8r.node.values)):
        a = get_exp_split(ds8r, index=i)
        for j in a:
            print(j.shape)

    a = get_exp_split(ds8r)
    for i in a:
        print(i.shape)

    print(len(ds8r.node.values))
