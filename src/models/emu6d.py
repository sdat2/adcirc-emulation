"""
Six dimensional emulation of the Holland 2008 model.
"""
import os
from typing import Callable, Tuple, Union
import shutil
from datetime import datetime, timedelta
import netCDF4 as nc
import numpy as np
import pandas as pd
import xarray as xr
from typeguard import typechecked
from omegaconf import OmegaConf

# WANDB_CACHE_DIR = "/work/n01/n01/sithom/tmp"
# WANDB_CONFIG_DIR = "/work/n01/n01/sithom/.config/wandb"
os.environ["WANDB_CACHE_DIR"] = "/work/n01/n01/sithom/tmp"
os.environ["WANDB_DATA_DIR"] = "/work/n01/n01/sithom/tmp"
os.environ["WANDB_DIR"] = "/work/n01/n01/sithom/.config/wandb"
os.environ["WANDB_CONFIG_DIR"] = "/work/n01/n01/sithom/.config/wandb"
os.environ["WANDB_MODE"] = "offline"  # "offline"
os.environ["MPLCONFIGDIR"] = "/work/n01/n01/sithom/.config/matplotlib"
import wandb
wandb.login(key="42ceaac64e4f3ae24181369f4c77d9ba0d1c64e5")
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import GPy
from GPy.kern.src.kern import Kern
from GPy.kern.src.stationary import Stationary
from GPy.kern import Linear, RBF, Matern32, Matern52
from GPy.models import GPRegression
from emukit.core.loop import OuterLoop
from emukit.core.acquisition import Acquisition
from emukit.bayesian_optimization.acquisitions import ExpectedImprovement
from emukit.bayesian_optimization.loops import BayesianOptimizationLoop
from emukit.experimental_design.experimental_design_loop import ExperimentalDesignLoop
from emukit.model_wrappers import SimpleGaussianProcessModel, GPyModelWrapper
from emukit.core.initial_designs.latin_design import LatinDesign
from emukit.experimental_design.acquisitions import ModelVariance
from emukit.bayesian_optimization.acquisitions import (
    MaxValueEntropySearch,
    ProbabilityOfImprovement,
    ExpectedImprovement,
)
from emukit.core import ParameterSpace, ContinuousParameter
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import imageio as io
from tqdm import tqdm, trange
from adcircpy.outputs import Maxele
from sithom.plot import plot_defaults, label_subplots
from sithom.time import timeit
from sithom.place import Point
from sithom.misc import in_notebook
from src.models.generation import ImpactSymmetricTC, Holland08
from src.constants import DATA_PATH, FIGURE_PATH, NEW_ORLEANS, NO_BBOX, CONFIG_PATH
from src.models.generation import vmax_from_pressure_holliday


@typechecked
def get_param(updates: dict) -> dict:
    """
    Get the parameters dictionary for the model (input to ADCIRC Holland Hurricane values).

    Args:
        updates (dict): The parameters that have been specified.

    Returns:
        dict: All required parameters, where those unspecified are set to Katrina's values.
    """
    # Default values are taken from a fit of the Holland 2008 model to Hurricane Katrina
    # At landfall.
    # These defaults should be loaded from config/katrina.yaml instead.
    defaults = {
        # Trajectory
        "angle": 0.0,  # degrees from North
        "speed": 7.71,  # m s**-1
        "point_east": 0.6,  # degrees East of New Orleans
        # Radial Profile of Tropical Cyclone - Holland Hurricane Parameters
        "rmax": 40744.0,  # meters
        "pc": 92800.0,  # Pa
        "vmax": 54.01667,  # m s**-1
        "xn": 1.1249,  # dimensionless
    }
    # no surprises allowed
    assert np.all([x in defaults.keys() for x in updates.keys()])

    # Is this necessary? There will be another origination for each call.
    output = defaults.copy()

    for key in updates:
        output[key] = updates[key]

    return output


def holliday_vmax(updates: dict) -> dict:
    """
    Generate vmax from pressure using Holliday's formula.

    Could be improved by finding vmax from the Holland 2008 model through pressure-gradient balance.

    Args:
        updates (dict): the dictionary of parameters, must contain "pc".

    Returns:
        dict: updates dictionary, to be passed to get_param.
    """
    assert "pc" in updates.keys()
    updates["vmax"] = vmax_from_pressure_holliday(updates["pc"])
    return updates


def get_data(ex_path: str) -> xr.Dataset:
    """
    Get the data from the ADCIRC simulation.

    # should we add in the parameters here?

    Args:
        ex_path (str): path to the ADCIRC simulation output folder.

    Returns:
        xr.Dataset: xarray dataset containing all of the data.
    """
    file_names = ["fort.73.nc", "fort.74.nc", "fort.63.nc", "fort.64.nc"]
    variables = [("pressure",), ("windx", "windy"), ("zeta",), ("u-vel", "v-vel")]
    ds_list = []
    traj_ds = xr.open_dataset(os.path.join(ex_path, "traj.nc"))
    for i in range(len(file_names)):
        for variable in variables[i]:
            ds_list.append(
                xr.Dataset(
                    data_vars={
                        variable: (
                            ["time", "node"],
                            nc.Dataset(os.path.join(ex_path, file_names[i]))[variable][
                                :
                            ],
                        ),
                    }
                )
            )
    merge_ds = xr.merge(ds_list)
    seconds_array = nc.Dataset(os.path.join(ex_path, file_names[0]))["time"][:]
    start_input = traj_ds["time"].values[0]
    start_input_date = datetime.utcfromtimestamp(
        (start_input - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, "s")
    )
    time_array = [
        start_input_date + timedelta(minutes=seconds_array[i] / 60 - 105 * 80)
        for i in range(len(seconds_array))
    ]
    merge_ds = merge_ds.assign_coords(time=time_array)
    x_array = nc.Dataset(os.path.join(ex_path, file_names[0]))["x"][:]
    y_array = nc.Dataset(os.path.join(ex_path, file_names[0]))["y"][:]
    depth_array = nc.Dataset(os.path.join(ex_path, file_names[0]))["depth"][:]
    element_array = nc.Dataset(os.path.join(ex_path, file_names[0]))["element"][:]
    mesh_ds = xr.Dataset(
        data_vars=dict(
            depth=(["node"], depth_array),
            triangle=(["element", "vertex"], element_array),
            x=(["node"], x_array),
            y=(["node"], y_array),
        )
    )
    file_names = ["maxele.63.nc", "maxwvel.63.nc", "maxvel.63.nc", "minpr.63.nc"]
    variables = [
        ("zeta_max", "time_of_zeta_max"),
        ("wind_max", "time_of_wind_max"),
        ("vel_max", "time_of_vel_max"),
        ("pressure_min", "time_of_pressure_min"),
    ]
    new_ds_list = []
    for i in range(len(file_names)):
        for variable in variables[i]:
            new_ds_list.append(
                xr.Dataset(
                    data_vars={
                        variable: (
                            ["node"],
                            nc.Dataset(os.path.join(ex_path, file_names[i]))[variable][
                                :
                            ],
                        ),
                    }
                )
            )
    max_ds = xr.merge(new_ds_list)
    return xr.merge(
        [
            traj_ds.rename({"time": "input_time", "lon": "clon", "lat": "clat"}),
            merge_ds.rename({"time": "output_time"}),
            mesh_ds.rename({"x": "lon", "y": "lat"}),
            max_ds,
        ]
    )


def real_func(param: dict, output_direc: str) -> float:
    """
    Feed the parameters into running the full tropical cyclone impact.

    Args:
        param (dict): _description_
        output_direc (str): Where to store data.

    Returns:
        float: The sea surface height at the point of interest.
    """
    wandb.init(
        project="6d_individual_version2",
        # settings=wandb.Settings(start_method="fork"),
        entity="sdat2",
        reinit=True,
        config=param,
    )
    point = Point(NEW_ORLEANS.lon + param["point_east"], NEW_ORLEANS.lat)
    if os.path.exists(output_direc):
        shutil.rmtree(output_direc)
    print("Running ImpactSymmetricTC", output_direc)
    ImpactSymmetricTC(
        point=point,
        output_direc=output_direc,
        symetric_model=Holland08(
            param["pc"], param["rmax"], param["vmax"], param["xn"]
        ),
        angle=param["angle"],
        trans_speed=param["speed"],
    ).run_impact()
    path = os.path.join(output_direc, "maxele.63.nc")
    maxele = Maxele(path, crs="EPSG:4326")
    places_d = dict(
        ansley=27,
        new_orleans=5,
        diamondhead=17,
        mississippi=77,
        atchafayala=82,
        dulac=86,
        akers=2,
    )
    indices = NO_BBOX.indices_inside(maxele.x, maxele.y)
    v = maxele.values[indices]
    height = v[places_d["ansley"]]
    print("height =  ", height, "m")
    wandb.log(
        {key: v[places_d[key]] for key in places_d},
    )
    combined_ds = get_data(output_direc)
    param_ds = xr.Dataset(data_vars=param)
    combined_ds = xr.merge([combined_ds, param_ds])
    combined_ds.to_netcdf(os.path.join(output_direc, "combined_ds.nc"))
    artifact = wandb.Artifact("output_dataset", type="dataset")
    artifact.add_file(os.path.join(output_direc, "combined_ds.nc"))
    wandb.log_artifact(artifact)
    return height


def fake_func(param: dict, output_direc: str) -> float:
    """
    A fake function to check the emulation would work.

    Args:
        param (dict): Parameters to be passed to the function.
        output_direc (str): Where to store data for this test.

    Returns:
        float: The mean of the absolute values of the parameters / 200.
    """
    default_param = get_param({})
    assert np.all([key in default_param.keys() for key in param])
    # print("called fake func")
    if os.path.exists(output_direc):
        shutil.rmtree(output_direc)
    if not os.path.exists(output_direc):
        os.mkdir(output_direc)

    return np.mean([abs(param[key]) for key in param]) / 20 / 100


class SixDOFSearch:
    """
    Six degrees of freedom search emulation.

    We could change the dimensions of the search space, but for now we will keep it as is.

    Data conversion example::
        >>> import numpy as np
        >>> tf = SixDOFSearch()
        >>> x_data = tf.samples(300)
        >>> np.all(np.isclose(tf.to_real(tf.to_normalized(x_data)), x_data, rtol=1e-6))
        True
        >>> np.all(tf.from_param(tf.to_param(x_data[0])) == x_data[0])
        True
    """

    def __init__(
        self,
        seed: int = 0,
        dryrun: bool = False,
        path: str = "6D_search",
        test_data_path: str = "6D_test",  # where to get the test data.
    ) -> None:
        """
        Initialize the search space for emulation.

        Args:
            seed (int, optional): Random seed (used by lhs). Defaults to 0.
            dryrun (bool, optional): Use a fake function. Defaults to False.
            path (str, optional): Where to store the data. Defaults to "6D_search".
            test_data_path (str, optional): Where to get the test data. Defaults to "6D_test".
        """
        np.random.seed(seed)
        # default ranges is from the sixd.yaml file
        conf = OmegaConf.load(os.path.join(CONFIG_PATH, "sixd.yaml"))
        self.dryrun = dryrun
        # angles = ContinuousParameter("angle", -90, 90)
        # speeds = ContinuousParameter("speed", 2, 14)
        # point_east = ContinuousParameter("point_east", -0.6, 1.2)
        # rmax = ContinuousParameter("rmax", 2000, 60000)
        # pc = ContinuousParameter("pc", 90000, 98000)
        # rmax = ContinuousParameter("rmax", 2, 14)
        # pc = ContinuousParameter("pc", 900, 980)
        # vmax = ContinuousParameter("vmax", 20)
        # xn = ContinuousParameter("xn", 0.8, 1.4)
        # set up all of the parameters automatically.
        self.space = ParameterSpace(
            [ContinuousParameter(i, conf[i].min, conf[i].max) for i in conf]
        )
        self.units = {i: conf[i].units for i in conf}
        # ['angle', 'speed', 'points_east', 'rmax', 'pc', 'xn']
        # self.space = ParameterSpace([angles, speeds, point_east, rmax, pc, xn])
        self.names = self.space.parameter_names
        self.normalized_space = ParameterSpace(
            [ContinuousParameter(name, 0, 1) for name in self.names]
        )
        # This seems somewhat redundant, but I guess I've included it for completeness.
        self.real_design = LatinDesign(self.space)
        # This latin hypercube design is used to generate the samples.
        self.normalized_design = LatinDesign(self.normalized_space)

        bounds = self.space.get_bounds()
        self.lower_bounds = np.asarray(bounds)[:, 0].reshape(1, len(bounds))
        self.upper_bounds = np.asarray(bounds)[:, 1].reshape(1, len(bounds))
        self.diffs = self.upper_bounds - self.lower_bounds
        self.call_number = 0

        # main figure paths.
        self.figure_path = os.path.join(FIGURE_PATH, path)
        self.data_path = os.path.join(DATA_PATH, path)
        self.test_data_path = os.path.join(DATA_PATH, test_data_path)

        # make file locations if they don't exist.
        for path in [self.figure_path, self.data_path]:
            if not os.path.exists(path):
                os.mkdir(path)

        # Setup-empty entries.
        self.init_x_data = np.array([[np.nan for _ in range(len(self.names))]])
        self.init_y_data = np.array([[np.nan]])
        self.active_x_data = np.array([[np.nan for _ in range(len(self.names))]])
        self.active_y_data = np.array([[np.nan]])
        self.test_x_data = np.array([[np.nan for _ in range(len(self.names))]])
        self.test_y_data = np.array([[np.nan]])

        # Load test data.
        #  self.load_test_data()

        # Data paths.
        self.x_path = os.path.join(self.data_path, "X.npy")
        self.y_path = os.path.join(self.data_path, "Y.npy")
        self.model_path = os.path.join(self.data_path, "model_save.npy")

    def __repr__(self) -> str:
        return f"SixDOFSearch({self.space}), units={self.units}), dryrun={self.dryrun})"

    def real_samples(self, num_samples: int) -> np.ndarray:
        return self.real_design.get_samples(num_samples).astype("float32")

    def normalized_samples(self, num_samples: int) -> np.ndarray:
        return self.normalized_design.get_samples(num_samples).astype("float32")

    def to_real(self, x_data: np.ndarray) -> np.ndarray:
        """
        x_data: assume last dimension is the variables.
        """
        ones = np.ones((x_data.shape[0], 1))
        return np.dot(ones, self.lower_bounds) + x_data * np.dot(ones, self.diffs)

    def to_normalized(self, x_data: np.ndarray) -> np.ndarray:
        """
        x_data: assume last dimension is the variables.
        """
        ones = np.ones((x_data.shape[0], 1))
        return (x_data - np.dot(ones, self.lower_bounds)) * np.dot(ones, 1 / self.diffs)

    def to_param(self, x_data_point: np.ndarray) -> dict:
        assert len(x_data_point) == len(self.names)
        assert len(np.shape(x_data_point)) == 1
        return holliday_vmax(
            {self.names[i]: x_data_point[i] for i in range(len(self.names))}
        )

    def from_param(self, param_dict: dict) -> np.ndarray:
        assert np.all([name in param_dict for name in self.names])
        output_np = np.zeros(len(self.names))
        for i, name in enumerate(self.names):
            output_np[i] = param_dict[name]
        return output_np

    def func(self, x_data: np.ndarray) -> np.ndarray:
        real_data = self.to_real(x_data)
        shape = np.shape(real_data)
        output_list = []

        # incorporate the tqdm progress bar if choosing more than one point.
        if shape[0] == 1:
            r = range(1)
        else:
            r = trange(shape[0], desc=f"Sweep, dryrun={self.dryrun}")

        for i in r:
            param = self.to_param(real_data[i])
            # print("Calling", param)
            output_direc = os.path.join(self.data_path, str(self.call_number))
            # print(output_direc)
            if self.dryrun:
                # take the negative for minimization.
                output_list.append(-fake_func(param, output_direc))
            else:
                # take the negative for minimization.
                output_list.append(-real_func(param, output_direc))
            self.call_number += 1

        return np.array(output_list).reshape(len(output_list), 1)

    def get_initial(self, samples: int = 500) -> None:
        # call func some number of times
        self.init_x_data = self.normalized_samples(samples)
        self.init_y_data = self.func(self.init_x_data)

    def _fit_initial(self, kernel_class: Union[Kern, Stationary] = Matern32):
        self.model_gpy = GPRegression(
            self.init_x_data,
            self.init_y_data.reshape(len(self.init_y_data), 1),
            kernel_class(6, 1),
        )
        self.model_gpy.optimize()
        self.model_emukit = GPyModelWrapper(self.model_gpy)

    def run_initial(
        self, samples: int = 500, kernel_class: Union[Kern, Stationary] = Matern32
    ) -> None:
        self.get_initial(samples=samples)
        self._fit_initial(kernel_class=kernel_class)

    def setup_active(
        self,
        acquisition_class: Acquisition = ModelVariance,
        loop_class: OuterLoop = BayesianOptimizationLoop,
    ) -> None:
        self.acquisition_function = acquisition_class(model=self.model_emukit)

        self.loop = loop_class(
            model=self.model_emukit,
            space=self.normalized_space,
            acquisition=self.acquisition_function,
            batch_size=1,
        )

        self.active_x_data = self.loop.loop_state.X[len(self.init_x_data) :]
        self.active_y_data = self.loop.loop_state.Y[len(self.init_x_data) :]

    def resetup_active(
        self,
        acquisition_class: Acquisition = ModelVariance,
        loop_class: OuterLoop = BayesianOptimizationLoop,
    ) -> None:
        # Add all data to the initial data variable.
        self.init_x_data = self.loop.loop_state.X
        self.init_y_data = self.loop.loop_state.Y
        if isinstance(acquisition_class, ModelVariance):
            self.acquisition_function = acquisition_class(model=self.model_emukit)
        # some acquisition functions require the space to be passed.
        elif isinstance(acquisition_class, MaxValueEntropySearch):
            self.acquisition_function = acquisition_class(
                model=self.model_emukit, space=self.normalized_space
            )

        self.loop = loop_class(
            model=self.model_emukit,
            space=self.normalized_space,
            acquisition=self.acquisition_function,
            batch_size=1,
        )
        # remove all data from the active data variable.
        self.active_x_data = self.loop.loop_state.X[len(self.init_x_data) :]
        self.active_y_data = self.loop.loop_state.Y[len(self.init_x_data) :]

    def run_active(self, new_iterations: int) -> None:
        self.loop.run_loop(self.func, new_iterations)
        self.active_x_data = self.loop.loop_state.X[len(self.init_x_data) :]
        self.active_y_data = self.loop.loop_state.Y[len(self.init_x_data) :]

    def save_as_normalized(self) -> None:
        np.save(self.x_path, self.loop.loop_state.X)
        np.save(self.y_path, self.loop.loop_state.Y)
        np.save(self.model_path, self.model_gpy.param_array)

    def load_data(self) -> None:
        # curently doesn't seem to load other data.
        # needs to run save_as_normalized first.
        X = np.load(self.x_path)
        Y = np.load(self.y_path)
        # model_param = np.load(self.model_path)
        # print(X, Y, model_param)
        self.save_normalized_to_netcdf(X, Y)

    def save_initial_data(self) -> None:
        # maybe this could just be called save_data?
        # won't work if setup action hasn't been run.
        X = self.loop.loop_state.X
        Y = self.loop.loop_state.Y
        # print(X, Y)
        self.save_normalized_to_netcdf(X, Y)

    def save_loop_data(self) -> None:
        X = self.init_x_data
        Y = self.init_y_data
        self.save_normalized_to_netcdf(X, Y)

    def save_normalized_to_netcdf(self, X, Y) -> None:
        x_real = self.to_real(X)
        y_real = -Y
        self.save_real(x_real, y_real)

    def save_real(self, x_real, y_real) -> None:

        # print(self.names)  # , x_real, y_real)
        points = list(range(x_real.shape[0]))
        num_vars = list(range(len(self.names)))
        # adding units would be good.
        # are the units in sixd.yaml?
        ds = xr.Dataset(
            data_vars={self.names[i]: (["point"], x_real[:, i]) for i in num_vars},
            coords=dict(
                point=points,
            ),
            attrs=dict(description="Data."),
        )
        # ds["maxele"] = y_real[:, 0]
        for i in [j for j in ds.variables if j in self.units]:
            ds[i].attrs = dict(units=self.units[i])

        ds = ds.assign(maxele=("point", y_real[:, 0]))
        ds.to_netcdf(os.path.join(self.data_path, "data.nc"))

    def load_real_data(self, data_path=None) -> xr.Dataset:
        if data_path is None:
            data_path = self.data_path
        # add option to use this for loading test data.
        return xr.open_dataset(os.path.join(data_path, "data.nc"))

    def load_real_df(self, data_path=None) -> pd.DataFrame:
        if data_path == None:
            data_path = self.data_path
        ds = self.load_real_data(data_path=data_path)
        df = ds.to_dataframe()
        return df.rename(
            columns={i: i + " [" + self.units[i] + "]" for i in self.names}
        )

    def load_normalized_data(self, data_path=None) -> Tuple[np.ndarray, np.ndarray]:
        print("loading data from", data_path)
        if data_path == None:
            data_path = self.data_path
        # load data from netcdf file and reconvert it to
        # normalised.
        ds = self.load_real_data(data_path=data_path)
        data = ds.to_array().values
        xr, yr = data[:-1], data[-1:]
        return self.to_normalized(xr.T), -yr.T

    def load_test_data(self, test_data_path=None) -> None:
        if test_data_path is None:
            test_data_path = self.test_data_path
        Xtest, Ytest = self.load_normalized_data(data_path=test_data_path)
        self.test_x_data = Xtest
        self.test_y_data = Ytest

    def test_metrics(self) -> None:
        # Test data need to be loaded first.
        mean, var = self.model_gpy.predict(self.test_x_data)
        rmse, mae, r2 = (
            mean_squared_error(self.test_y_data, mean, squared=False),
            mean_absolute_error(self.test_y_data, mean),
            r2_score(self.test_y_data, mean),
        )
        print("rmse", rmse, "mae", mae, "r2", r2)
        # check if wandb is running.
        if wandb.run is not None:
            wandb.log({"rmse": rmse, "mae": mae, "r2": r2})

    def gp_predict(self) -> Callable:
        X = np.load(self.x_path)
        Y = np.load(self.y_path)
        m_load = GPy.models.GPRegression(X, Y, initialize=False, kernel=Matern32(2, 1))
        m_load.update_model(
            False
        )  # do not call the underlying expensive algebra on load
        m_load.initialize_parameter()  # Initialize the parameters (connect the parameters up)
        m_load[:] = np.load(self.model_path)  # Load the parameters
        m_load.update_model(True)  # Call the algebra only once
        return m_load.predict

    def gp_predict_real(self) -> Callable:
        X = np.load(self.x_path)
        Y = np.load(self.y_path)
        m_load = GPy.models.GPRegression(X, Y, initialize=False, kernel=Matern32(2, 1))
        m_load.update_model(
            False
        )  # do not call the underlying expensive algebra on load
        m_load.initialize_parameter()  # Initialize the parameters (connect the parameters up)
        m_load[:] = np.load(self.model_path)  # Load the parameters
        m_load.update_model(True)  # Call the algebra only once

        def func_ret(x_data) -> Tuple[np.ndarray, np.ndarray]:
            mean, var = m_load.predict(self.to_normalized(x_data))
            return -mean, var

        return func_ret


def holdout_set() -> None:
    tf = SixDOFSearch(dryrun=False, path="6D_Search_Holdout", seed=5)
    print(tf.real_samples(100)[:10])
    print(tf.to_real(tf.normalized_samples(100))[:10])
    tf.run_initial(samples=200)
    tf.setup_active()
    tf.run_active(100)
    tf.save_normalized_to_netcdf()
    print(tf.gp_predict()(tf.normalized_samples(100)[:10]))
    print(tf.gp_predict_real()(tf.real_samples(100)[:10]))


def load_holdout_set() -> None:
    tf = SixDOFSearch(dryrun=False, path="6D_Search_Holdout", seed=5)
    tf.load_data()


def holdout_new() -> None:
    realholdout = SixDOFSearch(
        seed=105, dryrun=False, path="6D_Holdout", test_data_path="6DFake"
    )
    realholdout.run_initial(samples=1000)
    realholdout.setup_active()
    realholdout.save_initial_data()


def holdout_small(seed=2) -> None:
    realholdout = SixDOFSearch(
        seed=seed, dryrun=False, path="6D_Holdout_small", test_data_path="6DFake"
    )
    realholdout.run_initial(samples=250)
    realholdout.setup_active()
    realholdout.save_initial_data()


def test() -> None:
    tf = SixDOFSearch(dryrun=True, path="Test", seed=0)
    print(tf.real_samples(100)[:10])
    print(tf.to_real(tf.normalized_samples(100))[:10])
    tf.run_initial(samples=200)
    tf.setup_active()
    tf.run_active(100)
    tf.save_normalized_to_netcdf()
    print(tf.gp_predict()(tf.normalized_samples(100)[:10]))
    print(tf.gp_predict_real()(tf.real_samples(100)[:10]))


def holdout_tiny(seed=3) -> None:
    realholdout = SixDOFSearch(
        seed=seed,
        dryrun=False,
        path="6D_Holdout_tiny30",
        test_data_path="6DFake",
    )
    realholdout.run_initial(samples=80)
    realholdout.setup_active()
    realholdout.save_initial_data()


if __name__ == "__main__":
    # python src/models/emu6d.py
    # holdout_new()
    holdout_tiny(seed=30)
    # assert np.all(
    #    np.isclose(tf.real_samples(100), tf.to_real(tf.normalized_samples(100)), rtol=1e-3)
    # )
