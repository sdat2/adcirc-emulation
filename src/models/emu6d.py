import os
import shutil
import numpy as np
import pandas as pd
import xarray as xr
from frozendict import frozendict
import GPy
from GPy import kern
from GPy.kern import Linear, RBF, Matern32, Matern52
from GPy.models import GPRegression
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
from adcircpy.outputs import Maxele
from sithom.plot import plot_defaults, label_subplots
from sithom.time import timeit
from sithom.place import Point
from sithom.misc import in_notebook
from src.models.generation import ImpactSymmetricTC, Holland08
from src.constants import DATA_PATH, FIGURE_PATH, NEW_ORLEANS, NO_BBOX
from src.models.generation import vmax_from_pressure_holliday


def get_param(updates):
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
    # no surprises
    assert np.all([x in defaults.keys() for x in updates.keys()])

    output = defaults.copy()

    for key in updates:
        output[key] = updates[key]

    return output


def holliday_vmax(updates):
    assert "pc" in updates.keys()
    updates["vmax"] = vmax_from_pressure_holliday(92800)
    return updates


@np.vectorize
def indices_in_bbox(lon, lat):
    return (
        lon > NO_BBOX.lon[0]
        and lon < NO_BBOX.lon[1]
        and lat > NO_BBOX.lat[0]
        and lat < NO_BBOX.lat[1]
    )


def real_func(param, output_direc: str) -> float:
    point = Point(NEW_ORLEANS.lon + param["position_east"], NEW_ORLEANS.lat)
    if os.path.exists(output_direc):
        shutil.rmtree(output_direc)
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
    index_set = 27
    indices = indices_in_bbox(maxele.x, maxele.y)
    return maxele.values[indices][index_set]


def fake_func(param, output_direc: str) -> float:
    default_param = get_param({})
    assert np.all([key in default_param.keys() for key in param])
    print("called fake func")

    if os.path.exists(output_direc):
        shutil.rmtree(output_direc)

    if os.path.exists(output_direc):
        shutil.rmtree(output_direc)
    return 0.0


class TestFeature:
    """

    Data conversion example::
        >>> import numpy as np
        >>> tf = TestFeature()
        >>> x_data = tf.samples(300)
        >>> np.all(np.isclose(tf.to_real(tf.to_gp(x_data)), x_data, rtol=1e-6))
        True
        >>> np.all(tf.from_param(tf.to_param(x_data[0])) == x_data[0])
        True
    """

    def __init__(
        self,
        seed=0,
        dryrun=False,
        path="6D_search",
    ) -> None:
        np.random.seed(seed)
        self.dryrun = dryrun
        angles = ContinuousParameter("angle", -90, 90)
        speeds = ContinuousParameter("speed", 2, 14)
        point_east = ContinuousParameter("point_east", -0.6, 1.2)
        rmax = ContinuousParameter("rmax", 2, 14)
        pc = ContinuousParameter("pc", 900, 980)
        # vmax = ContinuousParameter("vmax", 20, )
        xn = ContinuousParameter("xn", 0.8, 1.4)
        self.space = ParameterSpace([angles, speeds, point_east, rmax, pc, xn])
        self.names = self.space.parameter_names
        self.gp_space = ParameterSpace(
            [ContinuousParameter(name, 0, 1) for name in self.names]
        )
        self.real_design = LatinDesign(self.space)
        self.gp_design = LatinDesign(self.gp_space)

        bounds = self.space.get_bounds()
        self.lower_bounds = np.asarray(bounds)[:, 0].reshape(1, len(bounds))
        self.upper_bounds = np.asarray(bounds)[:, 1].reshape(1, len(bounds))
        self.diffs = self.upper_bounds - self.lower_bounds
        self.call_number = 0

        # main figure paths.
        self.figure_path = os.path.join(FIGURE_PATH, path)
        self.data_path = os.path.join(DATA_PATH, path)

        # Setup-empty entries.
        self.init_x_data = np.array([[np.nan, np.nan]])
        self.init_y_data = np.array([[np.nan]])
        self.active_x_data = np.array([[np.nan, np.nan]])
        self.active_y_data = np.array([[np.nan]])

    def real_samples(self, num_samples: int) -> np.ndarray:
        return self.real_design.get_samples(num_samples).astype("float32")

    def gp_samples(self, num_samples: int) -> np.ndarray:
        return self.gp_design.get_samples(num_samples).astype("float32")

    def to_real(self, x_data: np.ndarray) -> np.ndarray:
        """
        x_data: assume last dimension is the variables.
        """
        ones = np.ones((x_data.shape[0], 1))
        return np.dot(ones, self.lower_bounds) + x_data * np.dot(ones, self.diffs)

    def to_gp(self, x_data: np.ndarray) -> np.ndarray:
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

        for i in range(shape[0]):
            param = self.to_param(real_data[i])
            print("Calling", param)
            output_direc = os.path.join(self.data_path, str(self.call_number))
            if self.dryrun:
                output_list.append(fake_func(param, output_direc))
            else:
                output_list.append(real_func(param, output_direc))
            self.call_number += 1

        return np.array(output_list).reshape(len(output_list), 1)

    def get_initial(self, samples=500):
        self.init_x_data = self.gp_samples(samples)
        self.init_y_data = self.func(self.init_x_data)

    def fit_initial(self, kernel_class=Matern32):
        self.model_gpy = GPRegression(
            self.init_x_data,
            self.init_y_data.reshape(len(self.init_y_data), 1),
            kernel_class(6, 1),
        )
        self.model_gpy.optimize()
        self.model_emukit = GPyModelWrapper(self.model_gpy)

    def run_initial(self, samples=500, kernel_class=Matern32) -> None:
        self.get_initial(samples=samples)
        self.fit_initial(kernel_class=kernel_class)

    def run_active(
        self, acquisition_class=ModelVariance, loop_class=BayesianOptimizationLoop
    ) -> None:
        self.acquisition_function = acquisition_class(model=self.model_emukit)

        self.loop = loop_class(
            model=self.model_emukit,
            space=self.gp_space,
            acquisition=self.acquisition_function,
            batch_size=1,
        )

        self.active_x_data = self.loop.loop_state.X[len(self.init_x_data) :]
        self.active_y_data = self.loop.loop_state.Y[len(self.init_x_data) :]


if __name__ == "__main__":
    # python src/models/emu6d.py
    tf = TestFeature(dryrun=True)
    print(tf.real_samples(100)[:10])
    print(tf.to_real(tf.gp_samples(100))[:10])
    tf.run_initial()
    tf.run_active()
    # assert np.all(
    #    np.isclose(tf.real_samples(100), tf.to_real(tf.gp_samples(100)), rtol=1e-3)
    # )
