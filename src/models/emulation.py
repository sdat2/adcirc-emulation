"""First implementation of EMUKIT emulation of ADCIRC model."""
from typing import List, Tuple
import os

os.environ["MPLCONFIGDIR"] = "/work/n01/n01/sithom/.config/matplotlib"
import shutil
import numpy as np
import pandas as pd
import xarray as xr
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
from src.constants import DATA_PATH, FIGURE_PATH, NEW_ORLEANS, NO_BBOX, PLACES_D


@np.vectorize
def indices_in_bbox(lon: float, lat: float) -> bool:
    return (
        lon > NO_BBOX.lon[0]
        and lon < NO_BBOX.lon[1]
        and lat > NO_BBOX.lat[0]
        and lat < NO_BBOX.lat[1]
    )


def example_plot() -> None:
    """
    Example plot.
    """

    plot_defaults()
    x_min = -30.0
    x_max = 30.0

    # x_data = np.random.uniform(x_min, x_max, (10, 1))
    p = ContinuousParameter("x", x_min, x_max)
    space = ParameterSpace([p])

    design = LatinDesign(space)
    num_data_points = 10
    x_data = design.get_samples(num_data_points)

    y_data = np.sin(x_data) + np.random.randn(10, 1) * 0.05
    emukit_model = SimpleGaussianProcessModel(x_data, y_data)

    loop = ExperimentalDesignLoop(space, emukit_model)
    loop.run_loop(np.sin, 30)
    plot_min = -40.0
    plot_max = 40.0
    real_x = np.arange(plot_min, plot_max, 0.2)
    real_y = np.sin(real_x)

    loop = ExperimentalDesignLoop(space, emukit_model)
    loop.run_loop(np.sin, 30)
    plot_min = -40.0
    plot_max = 40.0
    real_x = np.arange(plot_min, plot_max, 0.2)
    real_y = np.sin(real_x)
    predicted_y = []
    predicted_std = []
    for x in real_x:
        y, var = emukit_model.predict(np.array([[x]]))
        std = np.sqrt(var)
        predicted_y.append(y)
        predicted_std.append(std)

    predicted_y = np.array(predicted_y).flatten()
    predicted_std = np.array(predicted_std).flatten()
    plt.plot(real_x, predicted_y, label="estimated function", color="blue")
    plt.fill_between(
        real_x,
        predicted_y - predicted_std,
        predicted_y + predicted_std,
        color="blue",
        alpha=0.3,
    )
    plt.scatter(x_data, y_data, c="red")
    plt.plot(real_x, real_y, label="true function", color="red")
    plt.legend()

    if in_notebook():
        plt.show()
    else:
        plt.savefig(os.path.join(FIGURE_PATH, "emukit_example.png"))
        plt.clf()


def plot_space() -> None:
    """
    Make parameter space.
    """
    np.random.seed(0)
    plot_defaults()
    param_dict = frozendict(
        {
            "Direction [degrees]": (-70, 70),
            "Translation Speed [m s$^{-1}$]": (3, 10),
        }
    )

    param_list = [
        ContinuousParameter(param, param_dict[param][0], param_dict[param][1])
        for param in param_dict
    ]
    space = ParameterSpace(param_list)
    design = LatinDesign(space)
    num_init_data_points = 100
    x_data = design.get_samples(num_init_data_points)
    latin_cube = frozendict({param: x_data[:, i] for i, param in enumerate(param_dict)})
    params = list(latin_cube.keys())
    big_direc = os.path.join(DATA_PATH, "2dlhc")

    for i in range(num_init_data_points):
        os.path.join(big_direc, str(i), "maxele.63.nc")

    plt.scatter(
        latin_cube[params[0]], latin_cube[params[1]]
    )  # , c=latin_cube[params[2]])
    plt.xlabel(params[0])
    plt.ylabel(params[1])
    # plt.colorbar(label=params[2])
    plt.savefig(os.path.join(FIGURE_PATH, "2dlhc_param.png"))

    if in_notebook():
        plt.show()
    else:
        plt.clf()


def space() -> None:
    """
    Make parameter space.
    """
    param_dict = frozendict(
        {
            "Direction [degrees]": (-70, 70),
            "Speed [m s$^{-1}$]": (3, 10),
            # "Maximum velocity [m $^{-1}$]": (30, 60),
            # "Radius of maximum velocity [m]": (10e3, 50e3),
        }
    )

    param_list = [
        ContinuousParameter(param, param_dict[param][0], param_dict[param][1])
        for param in param_dict
    ]
    space = ParameterSpace(param_list)
    design = LatinDesign(space)
    num_init_data_points = 100
    x_data = design.get_samples(num_init_data_points)

    latin_cube = frozendict({param: x_data[:, i] for i, param in enumerate(param_dict)})

    params = list(latin_cube.keys())

    big_direc = os.path.join(DATA_PATH, "2dlhc")

    if not os.path.exists(big_direc):
        os.mkdir(big_direc)

    for i in range(num_init_data_points):
        ImpactSymmetricTC(
            # vmax=latin_cube[params[2]][i],
            trans_speed=latin_cube[params[1]][i],
            angle=latin_cube[params[0]][i],
            output_direc=os.path.join(big_direc, str(i)),
            symetric_model=Holland08(),
        ).run_impact()

    plt.scatter(latin_cube[params[0]], latin_cube[params[1]], c=latin_cube[params[2]])
    plt.xlabel(params[0])
    plt.ylabel(params[1])
    plt.colorbar(label=params[2])
    plt.savefig(os.path.join(FIGURE_PATH, "param.png"))

    if in_notebook():
        plt.show()
    else:
        plt.clf()
    # print(x_data)
    # print(x_data.shape)
    # print(space)
    # print(dir(space))
    # print(param_list)


def example_animation(tmp_dir: str = "tmp/") -> None:
    """
    Make an example animation with Latin Hypercube search.

    Args:
        tmp_dir (str, optional): temporary directory. Defaults to "tmp/".

    """

    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    plot_defaults()

    def func(x_inputs):
        return np.sin(x_inputs) + np.random.randn(len(x_inputs), 1) * 0.05

    x_min = -30.0
    x_max = 30.0
    y_min = -1.75
    y_max = -y_min
    var_max = 2.1
    var_min = 0

    x_param = ContinuousParameter("x", x_min, x_max)
    space = ParameterSpace([x_param])
    design = LatinDesign(space)
    num_init_data_points = 20
    x_data = design.get_samples(num_init_data_points)
    y_data = func(x_data)

    model_gpy = GPRegression(x_data, y_data)
    model_emukit = GPyModelWrapper(model_gpy)
    model_variance = ModelVariance(model=model_emukit)
    expdesign_loop = ExperimentalDesignLoop(
        model=model_emukit, space=space, acquisition=model_variance, batch_size=1
    )

    for i in range(10):
        plot_min = -40.0
        plot_max = 40.0
        real_x = np.arange(plot_min, plot_max, 0.2)
        real_y = np.sin(real_x)

        predicted_y = []
        predicted_std = []

        for x in real_x:
            y, var = model_emukit.predict(np.array([[x]]))
            std = np.sqrt(var)
            predicted_y.append(y)
            predicted_std.append(std)

        predicted_y = np.array(predicted_y).flatten()
        predicted_std = np.array(predicted_std).flatten()

        fig, (ax0, ax1) = plt.subplots(
            2, 1, gridspec_kw={"height_ratios": [3, 1]}, sharex=True
        )
        ax0.scatter(x_data, y_data, c="red", marker="+", label="initial data points")
        ax0.scatter(
            expdesign_loop.loop_state.X[len(x_data) :],
            expdesign_loop.loop_state.Y[len(x_data) :],
            c="red",
            label="new data points",
        )
        ax0.plot([x_min, x_min], [y_min, y_max], color="black")
        ax0.plot([x_max, x_max], [y_min, y_max], color="black")
        ax0.set_xlim([plot_min, plot_max])
        ax0.set_ylabel("y")
        ax0.plot(real_x, real_y, label="true function", color="red")
        ax0.set_ylim([y_min, y_max])
        ax0.legend(
            bbox_to_anchor=(0, 1.02, 1, 0.2),
            loc="lower left",
            mode="expand",
            borderaxespad=0,
            ncol=2,
        )
        ax0.fill_between(
            real_x,
            predicted_y - predicted_std,
            predicted_y + predicted_std,
            color="blue",
            alpha=0.3,
        )
        var = model_variance.evaluate(real_x.reshape(len(real_x), 1))
        ax1.plot(real_x, var)
        ax1.set_ylabel("Variance")
        ax1.plot([x_min, x_min], [var_min, var_max], color="black")
        ax1.plot([x_max, x_max], [var_min, var_max], color="black")
        ax1.set_xlabel("x")
        ax1.set_ylim([var_min, var_max])
        label_subplots([ax0, ax1])
        plt.savefig(os.path.join(tmp_dir, str(i) + ".png"), bbox_inches="tight")
        if in_notebook():
            plt.show()
        else:
            plt.clf()
        expdesign_loop.run_loop(func, 1)

    file_names = sorted((os.path.join(tmp_dir, fn) for fn in os.listdir(tmp_dir)))
    with io.get_writer(
        os.path.join("gifs", "max-var.gif"), mode="I", duration=0.5
    ) as writer:
        for filename in file_names:
            image = io.imread(filename)
            writer.append_data(image)
    writer.close()


@np.vectorize
def adcirc_func(
    angle: float, position: float, output_direc: str, index_set: int = 27
) -> float:
    """
    Run ADCIRC.

    Args:
        angle (float): Angle.
        position (float): Position east and west of New Orleans.
        output_direc (float): Where to store run.
        index_set (optional, int): Which height index to sample.

    Returns:
        float: maximum sea surface height at index_set.
    """
    point = Point(NEW_ORLEANS.lon + position, NEW_ORLEANS.lat)
    if os.path.exists(output_direc):
        shutil.rmtree(output_direc)
    ImpactSymmetricTC(
        point=point,
        output_direc=output_direc,
        symetric_model=Holland08(),
        angle=angle,
    ).run_impact()
    path = os.path.join(output_direc, "maxele.63.nc")
    maxele = Maxele(path, crs="EPSG:4326")
    indices = indices_in_bbox(maxele.x, maxele.y)
    return maxele.values[indices][index_set]


class EmulationBearingPos:
    """2D emulation of the ADCIRC model for bearing and position.
    Currently we're emulating the index point 27 in the maxele.63.nc
    file which is a point to the North of New Orleans
    This is the point with the highest
    storm surge in the 2005 Katrina event.
    """

    def __init__(
        self,
        seed: int = 0,
        init_num: int = 40,
        active_num: int = 30,
        indices: int = 100,
        acqusition_class=ExpectedImprovement,
        loop_class=BayesianOptimizationLoop,
        kernel_class=RBF,
        index: int = 27,
        x1_range: List[int] = [-90, 90],
        x2_range: List[int] = [-2, 3.2],
        path: str = "emu_angle_position",
    ) -> None:
        """Initialize the EmulationBearingPos class.

        Args:
            seed (int, optional): The seed for the random number generator.
            init_num (int, optional): The number of initial points to use for the emulation.
            active_num (int, optional): The number of active points to use for the emulation.
            indices (int): The number of indices to use for the emulation.
            acqusition_class (any, optional): The acquisition function to use for the emulation.
            loop_class (any, optional): The loop class to use for the emulation.
            kernel_class (any, optional): The kernel class to use for the emulation.
            index_set (int, optional)
            x1_range (List[int], optional): The range of the bearing parameter [degrees North].
            x2_range (List[int], optional): The range of the position parameter [degrees East].

        """
        # add option to use fake cheap function for testing
        self.seed = seed
        np.random.seed(seed)
        self.indices = indices
        x1_range, x2_range = self.to_gp_scale(np.array(x1_range), np.array(x2_range))
        x1_range = x1_range.tolist()
        x2_range = x2_range.tolist()
        self.ap = ContinuousParameter("a_param", *x1_range)
        self.bp = ContinuousParameter("b_param", *x2_range)
        self.space = ParameterSpace([self.ap, self.bp])
        self.design = LatinDesign(self.space)
        self.init_num = init_num
        self.active_num = active_num
        self.figure_path = os.path.join(FIGURE_PATH, path)
        self.data_path = os.path.join(DATA_PATH, path)
        self.call_number = 0
        self.acqusition_class = acqusition_class
        self.kernel_class = kernel_class
        self.loop_class = loop_class
        self.index = index

        for path in [self.figure_path, self.data_path]:
            if not os.path.exists(path):
                os.mkdir(path)

        # make empty arrays for data
        self.init_x_data = np.array([[np.nan, np.nan]])
        self.init_y_data = np.array([[np.nan]])
        self.active_x_data = np.array([[np.nan, np.nan]])
        self.active_y_data = np.array([[np.nan]])

        # could this not have been seperated into two or more functions?

    def setup_emulation(self) -> None:
        """Setup the emulation by running the
        function for the inital samples."""
        # run initial data.
        self.init_x_data = self.design.get_samples(self.init_num).astype("float32")
        self.init_y_data = self.func(self.init_x_data)

        # Fit initial GPyRegression
        self.model_gpy = GPRegression(
            self.init_x_data,
            self.init_y_data.reshape(len(self.init_y_data), 1),
            self.kernel_class(2, 1),
        )
        self.model_gpy.optimize()

        # active_learning - make acquisition file & loop.
        self.model_emukit = GPyModelWrapper(self.model_gpy)

        if self.acqusition_class is MaxValueEntropySearch:
            self.acquisition_function = self.acqusition_class(
                model=self.model_emukit, space=self.space
            )
        else:
            self.acquisition_function = self.acqusition_class(model=self.model_emukit)

        self.loop = self.loop_class(
            model=self.model_emukit,
            space=self.space,
            acquisition=self.acquisition_function,
            batch_size=1,
        )

    def active_learning(self) -> None:
        """Run the active learning loop for active_num.
        Setup emulation must have been run first."""

        # make initial plot
        self.plot()
        plt.savefig(os.path.join(self.figure_path, f"0.png"))
        if in_notebook():
            plt.show()
        else:
            plt.clf()

        for i in range(1, self.active_num + 1):
            print(i)
            self.run_loop(1)
            self.plot()
            self.save_data()
            plt.savefig(os.path.join(self.figure_path, f"{i}.png"))
            if in_notebook():
                plt.show()
            else:
                plt.clf()

        # make animation.
        with io.get_writer(f"{self.figure_path}.gif", mode="I", duration=0.5) as writer:
            for file_name in [
                os.path.join(self.figure_path, f"{i}.png")
                for i in range(self.active_num + 1)
            ]:
                image = io.imread(file_name)
                writer.append_data(image)
        writer.close()
        self.save_data()

    def save_data(self) -> None:
        """Save the sample data to a netcdf file so that it can be used later.
        The format is as follows:

        variables:
        init_x: The initial x data chosen by the latin hypercube design.
        init_y: The initial y data chosen by the latin hypercube design.
        active_x: The active x data chosen by the active learning loop.
        active_y: The active y data chosen by the active learning loop.

        coords:
            var: The variable names for the x data.
        """
        init_x, init_y = self.init_data()
        active_x, active_y = self.active_data()
        ds = xr.Dataset(
            data_vars=dict(
                init_x=(["inum", "var"], init_x),
                init_y=(["inum"], init_y[:, 0]),
                active_x=(["anum", "var"], active_x),
                active_y=("anum", active_y[:, 0]),
            ),
            coords=dict(
                var=(["var"], ["x1", "x2"]),
            ),
            attrs=dict(description="Training Data"),
        )
        file_name = os.path.join(self.data_path, "data.nc")
        if not os.path.exists(file_name):
            ds.to_netcdf(file_name)
        else:
            print("File Already Exists!")
            os.remove(file_name)
            ds.to_netcdf(file_name)

    def run_loop(self, new_iterations: int) -> None:
        """Run active learning loop."""
        self.loop.run_loop(self.func, new_iterations)
        self.active_x_data = self.loop.loop_state.X[len(self.init_x_data) :]
        self.active_y_data = self.loop.loop_state.Y[len(self.init_x_data) :]

    def init_data(self) -> Tuple[np.ndarray, np.ndarray]:
        return (
            self.merge(*self.to_real_scale(*self.split(self.init_x_data))),
            -self.init_y_data,
        )

    def active_data(self) -> Tuple[np.ndarray, np.ndarray]:
        return (
            self.merge(*self.to_real_scale(*self.split(self.active_x_data))),
            -self.active_y_data,
        )

    def __repr__(self) -> str:
        return str(
            f"seed = {self.seed}, init_num = {self.init_num}"
            + f"active_num = {self.active_num}"
        )

    def to_gp_scale(
        self, x1: np.ndarray, x2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return (x1 + 60) / 10, (x2 - 0.6) / 0.5

    def to_real_scale(
        self, x1: np.ndarray, x2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return x1 * 10 - 60, x2 / 2 + 0.6

    def merge(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [x1.reshape(*x1.shape, 1), x2.reshape(*x2.shape, 1)], axis=-1
        )

    def split(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        shape = data.shape
        if len(shape) == 2:
            return data[:, 0], data[:, 1]
        elif len(shape) == 3:
            return data[:, :, 0], data[:, :, 1]
        else:
            assert False

    def ob_adcirc_func(self, angle: np.array, position: float) -> float:
        # print(angle, position)
        num = len(angle)
        output_direc = [
            os.path.join(self.data_path, str(self.call_number + i)) for i in range(num)
        ]
        self.call_number += num
        return -adcirc_func(angle, position, output_direc, index_set=self.index)

    def func(self, data: np.ndarray) -> float:
        output = self.ob_adcirc_func(*self.to_real_scale(*self.split(data)))
        return output.reshape(len(output), 1)

    def learnt_function(self, x1, x2):
        # take real space inputs, get real space output of learnt function
        # Gaussian process eman and standard deviation.
        mean, var = self.model_emukit.predict(self.merge(*self.to_gp_scale(x1, x2)))
        return -mean, np.std(var)

    def plot(self) -> None:
        """Make a plot of the function, the sample data points, and the current
        values of the data acquisition function."""
        # indices are chosen for the colormaps
        a_indices = np.linspace(self.ap.min, self.ap.max, num=self.indices)
        b_indices = np.linspace(self.bp.min, self.bp.max, num=self.indices)
        a_indices, b_indices = self.to_real_scale(a_indices, b_indices)
        a_mesh, b_mesh = np.meshgrid(a_indices, b_indices)
        length = len(a_indices) * len(b_indices)
        a_array = a_mesh.ravel()
        b_array = b_mesh.ravel()
        comb_array = np.zeros([length, 2])
        comb_array[:, 0] = a_array[:]
        comb_array[:, 1] = b_array[:]
        comb_array_gp = self.merge(*self.to_gp_scale(*self.split(comb_array)))

        # Evaluate Gaussian Process
        mean, var = self.model_emukit.predict(comb_array_gp)
        mean_mesh = -mean[:, 0].reshape(self.indices, self.indices)
        std_mesh = np.sqrt(var[:, 0]).reshape(self.indices, self.indices)
        # Evaluate Acquisition Function
        aq_mesh = self.acquisition_function.evaluate(comb_array_gp).reshape(
            self.indices, self.indices
        )
        # let's save the data to a file incase we want to use it later
        xr.Dataset(
            data_vars=dict(
                mean=(["a", "b"], mean_mesh),
                std=(["a", "b"], std_mesh),
                aq=(["a", "b"], aq_mesh),
            ),
            coords=dict(a=(["a"], a_indices), b=(["b"], b_indices)),
        ).to_netcdf(
            os.path.join(
                self.data_path, "plotting_data" + str(self.call_number) + ".nc"
            )
        )
        # ok, now we have the data, let's plot it

        # Set up plot
        plot_defaults()
        fig, axs = plt.subplots(
            2, 2, sharex=True, sharey=True  # , figsize=get_dim(ratio=1)
        )
        label_subplots(axs, override="outside")

        ax = axs[0, 1]
        ax.set_title("Acqusition Function [m]")
        vmin = np.min(aq_mesh)
        vmax = np.max(aq_mesh)
        levels = np.linspace(vmin, vmax, num=400)
        im = ax.contourf(a_mesh, b_mesh, aq_mesh, vmin=vmin, vmax=vmax, levels=levels)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax, orientation="vertical")

        ax = axs[0, 0]
        init_x, init_y = self.init_data()
        active_x, active_y = self.active_data()

        cmap = plt.get_cmap("viridis")
        vmin = np.min(init_y)
        vmax = np.max(init_y)
        levels = np.linspace(vmin, vmax, num=400)

        im = ax.scatter(
            init_x[:, 0],
            init_x[:, 1],
            c=init_y,
            vmin=vmin,
            vmax=vmax,
            marker="x",
            label="Original data points",
        )
        ax.scatter(
            active_x[:, 0],
            active_x[:, 1],
            c=active_y,
            vmin=vmin,
            vmax=vmax,
            marker="+",
            label="New data points",
        )
        divider = make_axes_locatable(ax)
        ax.set_title("Samples [m]")
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax, orientation="vertical")
        ax.set_ylabel("Position [$^{\circ}$E]")

        ax = axs[1, 0]
        ax.set_title("Prediction Mean [m]")
        vmin = np.min(mean_mesh)
        vmax = np.max(mean_mesh)
        levels = np.linspace(vmin, vmax, num=400)
        im = ax.contourf(
            a_mesh, b_mesh, mean_mesh, vmin=vmin, vmax=vmax, levels=levels
        )  # , vmin=0, vmax=5.6)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax, orientation="vertical")
        ax.set_ylabel("Position [$^{\circ}$E]")
        ax.set_xlabel("Bearing [$^{\circ}$]")
        ax = axs[1, 1]
        ax.set_title("Prediction $\sigma$ [m]")
        vmin = np.min(std_mesh)
        vmax = np.max(std_mesh)
        levels = np.linspace(vmin, vmax, num=400)
        im = ax.contourf(a_mesh, b_mesh, std_mesh, vmin=vmin, vmax=vmax, levels=levels)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax, orientation="vertical")
        ax.set_xlabel("Bearing [$^{\circ}$]")


def plot():
    data = xr.open_dataset(
        os.path.join(DATA_PATH, "emulation_angle_pos_mves_no", "data.nc")
    )
    print(data)
    plot_defaults()

    ds_list = []
    for i in range(len(data.anum)):
        path = os.path.join(
            DATA_PATH, "emulation_angle_pos_mves_no", f"plotting_data{50+i}.nc"
        )
        ds = xr.open_dataset(path)
        ds_list.append(
            ds.expand_dims(dim="data_point").assign_coords({"data_point": [i]})
        )

    ds = xr.merge(ds_list)
    print(ds)
    vmin_mean = np.min(ds["mean"].values)
    vmax_mean = np.max(ds["mean"].values)
    vmin_std = np.min(ds["std"].values)
    vmax_std = np.max(ds["std"].values)
    vmin_aq = np.min(ds["aq"].values)
    vmax_aq = np.max(ds["aq"].values)
    a = ds.a.values
    b = ds.b.values
    a_mesh, b_mesh = np.meshgrid(a, b)
    # print(data.init_x, data.init_y)
    print(a_mesh.shape)
    print(ds["mean"].values.shape)

    def plot_index(index=0):
        # ok, now we have the data, let's plot it

        # Set up plot
        fig, axs = plt.subplots(
            2, 2, sharex=True, sharey=True  # , figsize=get_dim(ratio=1)
        )
        label_subplots(axs, override="outside")

        dsa = ds.isel(data_point=index)

        levels = np.linspace(vmin_aq, vmax_aq, num=400)
        ax = axs[0, 1]
        ax.set_title("Acqusition Function")
        im = ax.contourf(
            a_mesh, b_mesh, dsa.aq.values, vmin=vmin_aq, vmax=vmax_aq, levels=levels
        )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax, orientation="vertical")

        ax = axs[0, 0]
        cmap = plt.get_cmap("viridis")
        vmin = np.min(data.init_y)
        vmax = np.max(data.init_y)
        levels = np.linspace(vmin_mean, vmax_mean, num=400)
        im = ax.scatter(
            data.init_x[:, 0],
            data.init_x[:, 1],
            c=data.init_y,
            vmin=vmin_mean,
            vmax=vmax_mean,
            marker="x",
            label="Original data points",
        )
        ax.scatter(
            data.active_x[:index, 0],
            data.active_x[:index, 1],
            c=data.active_y[:index],
            vmin=vmin_mean,
            vmax=vmax_mean,
            marker="+",
            label="New data points",
        )
        divider = make_axes_locatable(ax)
        ax.set_title("Samples [m]")
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax, orientation="vertical")
        ax.set_ylabel("Position [$^{\circ}$E]")

        ax = axs[1, 0]
        ax.set_title("Prediction Mean [m]")
        levels = np.linspace(vmin_mean, vmax_mean, num=400)
        im = ax.contourf(
            a_mesh,
            b_mesh,
            dsa["mean"].values,
            vmin=vmin_mean,
            vmax=vmax_mean,
            levels=levels,
        )  # , vmin=0, vmax=5.6)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax, orientation="vertical")
        ax.set_ylabel("Position [$^{\circ}$E]")
        ax.set_xlabel("Bearing [$^{\circ}$]")
        ax = axs[1, 1]
        ax.set_title("Prediction $\sigma$ [m]")
        levels = np.linspace(vmin_std, vmax_std, num=400)
        im = ax.contourf(
            a_mesh,
            b_mesh,
            dsa["std"].values,
            vmin=vmin_std,
            vmax=vmax_std,
            levels=levels,
        )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax, orientation="vertical")
        ax.set_xlabel("Bearing [$^{\circ}$]")

    path = os.path.join(FIGURE_PATH, "gifs", "fig")
    os.makedirs(path, exist_ok=True)

    for i in range(30):
        plot_index(index=i)
        plt.savefig(os.path.join(path, str(i) + ".png"))
        plt.clf()


def animate():
    path = os.path.join("gifs", "fig")
    with io.get_writer(f"{path}.gif", mode="I", duration=0.5) as writer:
        for file_name in [os.path.join(path, f"{i}.png") for i in range(30)]:
            image = io.imread(file_name)
            writer.append_data(image)
    writer.close()


def poi():
    EmulationBearingPos(
        acqusition_class=ProbabilityOfImprovement, path="emulation_angle_pos_poi"
    )


def poi_long():
    EmulationBearingPos(
        acqusition_class=ProbabilityOfImprovement,
        path="emulation_angle_pos_poi_long",
        init_num=100,
        active_num=50,
    )


def mves():
    EmulationBearingPos(
        acqusition_class=MaxValueEntropySearch,
        path="emulation_angle_pos_mves",
        init_num=100,
        active_num=50,
        seed=101,
    )


def inp_diff():
    EmulationBearingPos(
        path="emulation_angle_pos_big",
        seed=20,
        init_num=100,
        active_num=30,
    )


def matern52():
    EmulationBearingPos(
        path="emulation_angle_pos_Mattern52",
        seed=30,
        init_num=100,
        active_num=30,
        kernel_class=Matern52,
    )


def matern32():
    e = EmulationBearingPos(
        path="emulation_angle_pos_Mattern32",
        seed=50,
        init_num=100,
        active_num=30,
        kernel_class=Matern32,
    )
    e.setup_emulation()
    e.active_learning()


def mat32():
    EmulationBearingPos(
        path="emulation_angle_pos_matern32_ent",
        seed=100,
        init_num=100,
        active_num=50,
        kernel_class=Matern32,
        acqusition_class=MaxValueEntropySearch,
    )


def mat32var():
    EmulationBearingPos(
        path="emulation_angle_pos_matern32_variance",
        seed=100,
        init_num=200,
        active_num=50,
        kernel_class=Matern32,
        acqusition_class=ModelVariance,
    )


def mat32expimprovement():
    e = EmulationBearingPos(
        path="emulation_angle_pos_newei7",
        seed=107,
        init_num=100,
        active_num=30,
        kernel_class=Matern32,
        acqusition_class=ExpectedImprovement,
    )
    e.setup_emulation()
    e.active_learning()


def mat32expimprovementno():
    e = EmulationBearingPos(
        path="emulation_angle_pos_ei_no",
        seed=107,
        init_num=50,
        active_num=30,
        index=PLACES_D["new_orleans"],
        kernel_class=Matern32,
        acqusition_class=ExpectedImprovement,
    )
    e.setup_emulation()
    e.active_learning()


def mat32mvesno():
    e = EmulationBearingPos(
        path="emulation_angle_pos_mves_no",
        seed=107,
        init_num=50,
        active_num=30,
        index=PLACES_D["new_orleans"],
        kernel_class=Matern32,
        acqusition_class=MaxValueEntropySearch,
    )
    e.setup_emulation()
    e.active_learning()


def test():
    e = EmulationBearingPos(
        path="emulation_angle_pos_t3",
        seed=2,
        init_num=1,
        active_num=1,
        kernel_class=Matern32,
        acqusition_class=ExpectedImprovement,
    )
    e.setup_emulation()
    e.active_learning()
    # e.save_initial_data()


if __name__ == "__main__":
    # python src/models/emulation.py
    # example_animation()
    # example_plot()
    plot_defaults()
    # mat32mvesno()
    # plot()
    animate()
    # mves()
    # test()
