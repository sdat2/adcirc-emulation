"""
Simple timeseries models.
"""

import os
from typing import Tuple, List
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import wandb
import omegaconf
from omegaconf import OmegaConf, DictConfig
from uncertainties import ufloat
import hydra
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sithom.time import timeit
from sithom.place import BoundingBox
from sithom.plot import plot_defaults
from src.constants import CONFIG_PATH, DATA_PATH, NO_BBOX, FIGURE_PATH

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, LSTM

# FEATURE_LIST = [#
FEATURE_LIST: List[str] = ["angle", "speed", "point_east", "rmax", "pc", "xn"]
DEFAULT_INDEX = 27
ART_PATH = os.path.join(DATA_PATH, "artifacts")
os.makedirs(ART_PATH, exist_ok=True)
# weird mac issue
# DATA_PATH = "Users/simon/new-orleans/data"


def make_8d_data(
    num: int = 200, bbox: BoundingBox = NO_BBOX, filter_by_bbox: bool = False
) -> None:
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
        print(f"artifact: {version}")
        artifact_dir = artifact.download(root=ART_PATH)
        cds_a = xr.open_dataset(os.path.join(artifact_dir, "combined_ds.nc"))
        print("cds_a", cds_a)
        print(cds_a.data_vars)

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
        oa = cds_a[
            [
                "zeta",
                "u-vel",
                "v-vel",
                "windx",
                "windy",
                "pressure",
                "triangle",
                "depth",
                "lon",
                "lat",
            ]
        ].interp({"output_time": cds_a["input_time"]})
        if filter_by_bbox:
            indices = bbox.indices_inside(cds_a["lon"].values, cds_a["lat"].values)
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


if "__main__" == __name__:
    # python src/models/simple_timeseries.py
    make_8d_data(num=100)
    pass


@timeit
def load_8d_data() -> xr.Dataset:
    """
    Load the 8d dataset.

    Returns:
        xr.Dataset: xarray dataset.
    """
    print("PATH:", os.path.join(DATA_PATH, "ds8.nc"))
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


def descale_x(x: np.ndarray, ds: xr.Dataset, node: int = DEFAULT_INDEX) -> np.ndarray:
    print(ds)
    print(x.shape)
    mu_x = ds.mu_x.values
    sigma_x = ds.sigma_x.values
    xn = np.copy(x)
    for i in range(len(mu_x)):
        xn[:, i] = xn[:, i] * sigma_x[i] + mu_x[i]
    return xn


def descale_y(y: np.ndarray, ds: xr.Dataset, node: int = DEFAULT_INDEX) -> np.ndarray:
    print("descale")
    print(y.shape)
    print(ds)
    mu_y = ds.mu_y.isel(node=node).values[0]
    sigma_y = ds.sigma_y.isel(node=node).values[0]
    return y * sigma_y + mu_y


@timeit
def get_simple_split(
    ds8r: xr.Dataset, index: int = 26
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


@timeit
def train_mlp(x_train: np.ndarray, y_train: np.ndarray, seed: int = 0) -> any:
    """
    Train a multi-layer perceptron with x_train, y_train.

    Args:
        x_train (np.ndarray): x training data.
        y_train (np.ndarray): y training data.
        seed (int): random seed to initialize weights.

    Returns:
        any: Trained multi layer perceptron model.
    """
    np.random.seed(seed)
    model = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=500).fit(
        x_train, y_train
    )
    print("loss", model.loss_)

    return model


def pred_mlp(x_test: np.ndarray, model: any) -> np.ndarray:
    """
    Test a multi-layer perceptron with x_test, model.

    Args:
        x_test (np.ndarray): x test data.
        model (any): trained model.

    Returns:
        np.ndarray: predictions.
    """
    predictions = model.predict(x_test)
    return predictions


@timeit
def single_model_pred() -> None:
    ds8r = rescale_ds(load_8d_data())

    # Get Experiment Split..
    x_train, x_test, y_train, y_test = get_exp_split(
        ds8r, index=DEFAULT_INDEX, split_index=150
    )
    for i in (x_train, x_test, y_train, y_test):
        # x_train, x_test, y_train, y_test
        print(i.shape)

    print(len(ds8r.node.values))
    # train a model
    model = train_mlp(x_train, y_train)
    # test a model
    y_pred = pred_mlp(x_test, model)
    print(y_pred)
    # xtst = descale_x(x_test, ds8r, node=DEFAULT_INDEX)
    ytst = descale_y(y_test, ds8r, node=DEFAULT_INDEX)
    ypr = descale_y(y_pred, ds8r, node=DEFAULT_INDEX)

    score = model.score(x_test, y_test)
    # descale(x, y, ds8r)
    plot_defaults()
    ymin = min(min(ytst), min(ypr))
    ymax = max(max(ytst), max(ypr))
    plt.plot([ymin, ymax], [ymin, ymax], color="black")
    plt.scatter(ytst, ypr, s=2)
    plt.text(
        ymin, ymax - 0.5, "$r^2=${:.3f}".format(score) + ",  n={:}".format(len(ypr))
    )
    plt.xlabel("Physical Model, SSH [m]")
    plt.ylabel("Statistical Model, SSH [m]")
    plt.savefig(os.path.join(FIGURE_PATH, "pred.png"))
    plt.clf()


def ensemble_pred(ensemble_size: int = 30, index: int = DEFAULT_INDEX) -> ufloat:
    ds8r = rescale_ds(load_8d_data())

    # Get Experiment Split.
    x_train, x_test, y_train, y_test = get_exp_split(ds8r, index=index, split_index=150)
    for i in (x_train, x_test, y_train, y_test):
        # x_train, x_test, y_train, y_test
        print(i.shape)

    print(len(ds8r.node.values))
    # train a model
    models = [train_mlp(x_train, y_train, seed=x) for x in range(ensemble_size)]
    # test a model
    y_preds = [pred_mlp(x_test, model) for model in models]
    # print(y_pred)
    xtst = descale_x(x_test, ds8r, node=DEFAULT_INDEX)
    ytst = descale_y(y_test, ds8r, node=DEFAULT_INDEX)
    yprs = [descale_y(y_pred, ds8r, node=DEFAULT_INDEX) for y_pred in y_preds]
    scores = [model.score(x_test, y_test) for model in models]
    print(scores)
    # descale(x, y, ds8r)
    plot_defaults()
    ymin = min(min(ytst), min([min(ypr) for ypr in yprs]))
    ymax = max(max(ytst), max([max(ypr) for ypr in yprs]))
    plt.plot([ymin, ymax], [ymin, ymax], color="black")
    plt.scatter(ytst, yprs[0], s=2)
    plt.text(
        ymin,
        ymax - 0.5,
        "$r^2=${:.3f}".format(scores[0]) + ",  n={:}".format(len(ytst)),
    )
    plt.xlabel("Physical Model, SSH [m]")
    plt.ylabel("Statistical Model, SSH [m]")
    plt.savefig(os.path.join(FIGURE_PATH, "pred.png"))
    plt.clf()
    plt.plot([ymin, ymax], [ymin, ymax], color="black")
    plt.text(
        ymin,
        ymax - 0.5,
        "$r^2=${:.3f}".format(np.mean(scores))
        + "$\pm${:.3f}".format(np.std(scores))
        + ",  n={:}".format(len(ytst)),
    )

    yprs = np.array(yprs)
    print(yprs.shape)
    # plot predictions
    mn = np.mean(yprs, axis=0)
    std = np.std(yprs, axis=0)
    plt.errorbar(ytst, mn, yerr=std, color="green", fmt="x")
    plt.scatter(ytst, np.mean(yprs, axis=0), s=2)
    plt.xlabel("Physical Model, SSH [m]")
    plt.ylabel("Statistical Model, SSH [m]")
    plt.savefig(os.path.join(FIGURE_PATH, "ensemble_pred.png"))
    plt.clf()
    return ufloat(np.mean(scores), np.std(scores))


def train_lstm(
    x_train: np.array,
    y_train: np.array,
    look_back: int = 1,
    epochs: int = 10,
    batch_size: int = 1,
):
    """Train an LSTM model."""
    model = Sequential()  # Sequential Keras Model
    model.add(LSTM(4, input_shape=(8, look_back)))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam")
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)
    return model


def lstm_data_loader(
    split_index: int = 150, index: int = DEFAULT_INDEX, look_back: int = 1
):
    """Load data into format needed for LSTM setup, for single point timeseries.

    The previous timesteps are added as an additional dimension, length look_back.

    Args:
        split_index (int): Where to split the experiments. Defaults to 150.
        index (int): Which node to choose to model. Default to 27.
        look_back (int): How long to allow LSTM to look back.

    Returns:
        Tuple[np.array, np.array, np.array, np.array]: x_train, x_test, y_train, y_test.
    """

    def create_dataset(xt, yt, look_back=look_back):
        dataX, dataY = [], []
        for i in range(len(xt) - look_back - 1):
            dataX.append(xt[i : (i + look_back), 0])
            dataY.append(yt[i + look_back, 0])
        return np.array(dataX), np.array(dataY)

    ds8r = rescale_ds(load_8d_data())
    ex8d = ds8r.isel(node=index, exp=slice(0, split_index), output=0)
    x_values = ex8d.x.values
    y_values = ex8d.y.values
    x_train = x_values
    y_train = y_values
    # x_train = x_values.reshape(-1, x_values.shape[-1])
    # y_train = np.nan_to_num(y_values.reshape(-1, y_values.shape[-1]).ravel())
    x_train, y_train = create_dataset(x_train, y_train)
    ex8d = ds8r.isel(node=index, exp=slice(split_index, 286), output=0)
    x_values = ex8d.x.values
    y_values = ex8d.y.values
    x_test = x_values
    y_test = y_values
    # x_test = x_values.reshape(-1, x_values.shape[-1])
    # y_test = np.nan_to_num(y_values.reshape(-1, y_values.shape[-1]).ravel())
    x_test, y_test = create_dataset(x_test, y_test)

    return x_train, x_test, y_train, y_test


if "__main__" == __name__:
    # python src/models/simple_timeseries.py
    # make_8d_data()
    # for a in lstm_data_loader():
    #     print(a.shape)
    pass


def other():
    ds8r = rescale_ds(load_8d_data())

    if False:
        # go through all of the possible nodes.
        for i in range(len(ds8r.node.values)):
            x_train, x_test, y_train, y_test = get_exp_split(
                ds8r, index=i, split_index=150
            )
            for j in (x_train, x_test, y_train, y_test):
                print(j.shape)

    # Get Experiment Split..
    x_train, x_test, y_train, y_test = get_exp_split(
        ds8r, index=DEFAULT_INDEX, split_index=150
    )
    for i in (x_train, x_test, y_train, y_test):
        # x_train, x_test, y_train, y_test
        print(i.shape)

    print(len(ds8r.node.values))
    # train a model
    models = [train_mlp(x_train, y_train, seed=x) for x in range(30)]
    # test a model
    y_preds = [pred_mlp(x_test, model) for model in models]
    # print(y_pred)
    xtst = descale_x(x_test, ds8r, node=DEFAULT_INDEX)
    ytst = descale_y(y_test, ds8r, node=DEFAULT_INDEX)
    yprs = [descale_y(y_pred, ds8r, node=DEFAULT_INDEX) for y_pred in y_preds]
    scores = [model.score(x_test, y_test) for model in models]
    # descale(x, y, ds8r)
    plot_defaults()
    ymin = min(min(ytst), min([min(ypr) for ypr in yprs]))
    ymax = max(max(ytst), max([max(ypr) for ypr in yprs]))
    plt.plot([ymin, ymax], [ymin, ymax], color="black")
    plt.scatter(ytst, yprs[0], s=2)
    plt.text(
        ymin,
        ymax - 0.5,
        "$r^2=${:.3f}".format(scores[0]) + ",  n={:}".format(len(ytst)),
    )
    plt.xlabel("Physical Model, SSH [m]")
    plt.ylabel("Statistical Model, SSH [m]")
    plt.savefig(os.path.join(FIGURE_PATH, "pred.png"))
    plt.clf()
    plt.plot([ymin, ymax], [ymin, ymax], color="black")
    plt.text(
        ymin,
        ymax - 0.5,
        "$r^2=${:.3f}".format(np.mean(scores))
        + "$\pm${:.3f}".format(np.std(scores))
        + ",  n={:}".format(len(ytst)),
    )

    yprs = np.array(yprs)
    print(yprs.shape)
    mn = np.mean(yprs, axis=0)
    std = np.std(yprs, axis=0)
    plt.errorbar(ytst, mn, yerr=std, color="green", fmt="x")
    plt.scatter(ytst, np.mean(yprs, axis=0), s=2)
    plt.xlabel("Physical Model, SSH [m]")
    plt.ylabel("Statistical Model, SSH [m]")
    plt.savefig(os.path.join(FIGURE_PATH, "ensemble_pred.png"))
    plt.clf()
