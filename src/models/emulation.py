"""EMUKIT"""
import os
from GPy import kern
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio as io
from frozendict import frozendict
import GPy
from GPy.models import GPRegression
from emukit.experimental_design.experimental_design_loop import ExperimentalDesignLoop
from emukit.model_wrappers import SimpleGaussianProcessModel
from emukit.core import ParameterSpace, ContinuousParameter
from emukit.model_wrappers import GPyModelWrapper
from emukit.experimental_design.acquisitions import ModelVariance
from sithom.plot import plot_defaults, label_subplots
from sithom.misc import in_notebook
from src.constants import DATA_PATH, FIGURE_PATH
from src.models.generation import ImpactSymmetricTC, Holland08
from emukit.core.initial_designs.latin_design import LatinDesign
import shutil
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from adcircpy.outputs import Maxele
import imageio as io
from frozendict import frozendict
import GPy
from GPy.kern import Linear, RBF, Matern32, Matern52
from GPy.models import GPRegression
from emukit.bayesian_optimization.acquisitions import ExpectedImprovement
from emukit.bayesian_optimization.loops import BayesianOptimizationLoop
from emukit.experimental_design.experimental_design_loop import ExperimentalDesignLoop
from emukit.model_wrappers import SimpleGaussianProcessModel
from emukit.core import ParameterSpace, ContinuousParameter
from emukit.core.initial_designs.latin_design import LatinDesign
from emukit.model_wrappers import GPyModelWrapper
from emukit.experimental_design.acquisitions import ModelVariance
from emukit.bayesian_optimization.acquisitions import (
    MaxValueEntropySearch,
    ProbabilityOfImprovement,
    ExpectedImprovement,
)
from sithom.plot import plot_defaults, label_subplots
from sithom.time import timeit
from sithom.place import Point
from sithom.misc import in_notebook
from src.constants import DATA_PATH, FIGURE_PATH
from src.models.generation import ImpactSymmetricTC, Holland08
from src.constants import NEW_ORLEANS, DATA_PATH, NO_BBOX


@np.vectorize
def indices_in_bbox(lon, lat):
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
def smash_func(angle: float, position: float, output_direc: str) -> float:
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
    index_set = 27
    indices = indices_in_bbox(maxele.x, maxele.y)
    return maxele.values[indices][index_set]


class EmulationSmash:
    def __init__(
        self,
        seed=0,
        init_num=40,
        active_num=30,
        indices=100,
        acqusition_class=ExpectedImprovement,
        loop_class=BayesianOptimizationLoop,
        kernel_class=RBF,
        x1_range=[-90, 90],
        x2_range=[-2, 3.2],
        path="emu_angle_position",
    ) -> None:
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

        for path in [self.figure_path, self.data_path]:
            if not os.path.exists(path):
                os.mkdir(path)

        # run initial data.
        self.init_x_data = self.design.get_samples(self.init_num).astype("float32")
        self.init_y_data = self.func(self.init_x_data)

        # Fit initial GPyRegression
        self.model_gpy = GPRegression(
            self.init_x_data,
            self.init_y_data.reshape(len(self.init_y_data), 1),
            kernel_class(2, 1),
        )
        self.model_gpy.optimize()

        # active_learning - make acquisition file & loop.
        self.model_emukit = GPyModelWrapper(self.model_gpy)
        if acqusition_class is MaxValueEntropySearch:
            self.acquisition_function = acqusition_class(
                model=self.model_emukit, space=self.space
            )
        else:
            self.acquisition_function = acqusition_class(model=self.model_emukit)

        self.loop = loop_class(
            model=self.model_emukit,
            space=self.space,
            acquisition=self.acquisition_function,
            batch_size=1,
        )

        # get ready for active learning.
        self.active_x_data = np.array([[np.nan, np.nan]])
        self.active_y_data = np.array([[np.nan]])

        # make initial plot
        self.plot()
        plt.savefig(os.path.join(self.figure_path, f"0.png"))
        if in_notebook():
            plt.show()
        else:
            plt.clf()

        for i in range(1, active_num + 1):
            print(i)
            self.run_loop(1)
            self.plot()
            plt.savefig(os.path.join(self.figure_path, f"{i}.png"))
            if in_notebook():
                plt.show()
            else:
                plt.clf()

        with io.get_writer(f"{self.figure_path}.gif", mode="I", duration=0.5) as writer:
            for file_name in [
                os.path.join(self.figure_path, f"{i}.png")
                for i in range(self.active_num + 1)
            ]:
                image = io.imread(file_name)
                writer.append_data(image)
        writer.close()
        self.save_data()

    def save_data(self):
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
            ds.to_netcdf(os.path.join(self.data_path, "data.nc"))
        else:
            print("File Already Exists!")

    def run_loop(self, new_iterations):
        self.loop.run_loop(self.func, new_iterations)
        self.active_x_data = self.loop.loop_state.X[len(self.init_x_data) :]
        self.active_y_data = self.loop.loop_state.Y[len(self.init_x_data) :]

    def init_data(self):
        return (
            self.merge(*self.to_real_scale(*self.split(self.init_x_data))),
            -self.init_y_data,
        )

    def active_data(self):
        return (
            self.merge(*self.to_real_scale(*self.split(self.active_x_data))),
            -self.active_y_data,
        )

    def __repr__(self) -> str:
        return f"seed = {self.seed}, init_num = {self.init_num}, active_num = {self.active_num}"

    def to_gp_scale(self, x1, x2):
        return (x1 + 60) / 10, (x2 - 0.6) / 0.5

    def to_real_scale(self, x1, x2):
        return x1 * 10 - 60, x2 / 2 + 0.6

    def merge(self, x1, x2):
        return np.concatenate(
            [x1.reshape(*x1.shape, 1), x2.reshape(*x2.shape, 1)], axis=-1
        )

    def split(self, data):
        shape = data.shape
        if len(shape) == 2:
            return data[:, 0], data[:, 1]
        elif len(shape) == 3:
            return data[:, :, 0], data[:, :, 1]
        else:
            assert False

    def ob_smash_func(self, angle: np.array, position: float) -> float:
        # print(angle, position)
        num = len(angle)
        output_direc = [
            os.path.join(self.data_path, str(self.call_number + i)) for i in range(num)
        ]
        self.call_number += num
        return -smash_func(angle, position, output_direc)

    def func(self, data) -> float:
        output = self.ob_smash_func(*self.to_real_scale(*self.split(data)))
        return output.reshape(len(output), 1)

    def learnt_function(self, x1, x2):
        mean, var = self.model_emukit.predict(self.merge(*self.to_gp_scale(x1, x2)))
        return -mean, np.std(var)

    def plot(self) -> None:
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

        # Set up plot
        plot_defaults()
        fig, axs = plt.subplots(
            2, 2, sharex=True, sharey=True  # , figsize=get_dim(ratio=1)
        )
        label_subplots(axs, override="outside")

        ax = axs[0, 1]
        ax.set_title("Acq. Func.")
        im = ax.contourf(a_mesh, b_mesh, aq_mesh)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax, orientation="vertical")

        ax = axs[0, 0]
        init_x, init_y = self.init_data()
        active_x, active_y = self.active_data()
        im = ax.scatter(
            init_x[:, 0],
            init_x[:, 1],
            c=init_y,
            marker="x",
            label="original data points",
        )
        ax.scatter(
            active_x[:, 0],
            active_x[:, 1],
            c=active_y,
            marker="+",
            label="new data points",
        )
        divider = make_axes_locatable(ax)
        ax.set_title("Samples")
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax, orientation="vertical")
        ax.set_ylabel("Position [$^{\circ}$E]")

        ax = axs[1, 0]
        ax.set_title("Prediction Mean")
        im = ax.contourf(a_mesh, b_mesh, mean_mesh)  # , vmin=0, vmax=5.6)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax, orientation="vertical")
        ax.set_ylabel("Position [$^{\circ}$E]")
        ax.set_xlabel("Bearing [$^{\circ}$]")
        ax = axs[1, 1]
        ax.set_title("Pred. Std. Dev. ")
        im = ax.contourf(a_mesh, b_mesh, std_mesh)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax, orientation="vertical")
        ax.set_xlabel("Bearing [$^{\circ}$]")


def poi():
    EmulationSmash(
        acqusition_class=ProbabilityOfImprovement, path="emulation_angle_pos_poi"
    )


def poi_long():
    EmulationSmash(
        acqusition_class=ProbabilityOfImprovement,
        path="emulation_angle_pos_poi_long",
        init_num=100,
        active_num=50,
    )


def mves():
    EmulationSmash(
        acqusition_class=MaxValueEntropySearch, path="emulation_angle_pos_mves"
    )


def inp_diff():
    EmulationSmash(
        path="emulation_angle_pos_big",
        seed=20,
        init_num=100,
        active_num=30,
    )


def mattern52():
    EmulationSmash(
        path="emulation_angle_pos_Mattern52",
        seed=30,
        init_num=100,
        active_num=30,
        kernel_class=Matern52
    )

def mattern32():
    EmulationSmash(
        path="emulation_angle_pos_Mattern32",
        seed=50,
        init_num=100,
        active_num=30,
        kernel_class=Matern32
    )


if __name__ == "__main__":
    # python src/models/emulation.py
    # example_animation()
    # example_plot()
    plot_defaults()
    mattern52()
