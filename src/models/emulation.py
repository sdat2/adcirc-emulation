"""EMUKIT"""
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio as io
import GPy
from GPy.models import GPRegression
from emukit.experimental_design.experimental_design_loop import ExperimentalDesignLoop
from emukit.model_wrappers import SimpleGaussianProcessModel
from emukit.core import ParameterSpace, ContinuousParameter
from emukit.model_wrappers import GPyModelWrapper
from emukit.experimental_design.acquisitions import ModelVariance
from sithom.plot import plot_defaults, label_subplots
from sithom.misc import in_notebook
from src.constants import FIGURE_PATH
from emukit.core.initial_designs.latin_design import LatinDesign


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


def space() -> None:
    """
    Make parameter space.
    """
    dir = ContinuousParameter("dir [degrees]", -70, 70)
    trans_speed = ContinuousParameter("Speed [m s**-1]", 1, 5)
    vmax = ContinuousParameter("Maximum velocity [m s**-1]", 20, 60)
    rmax = ContinuousParameter("Radius of maximum velocity [m]", 10e3, 100e3)
    bs = ContinuousParameter("bs [?dim?]", 6, 8)
    space = ParameterSpace([dir, trans_speed, vmax, rmax, bs])
    design = LatinDesign(space)
    num_init_data_points = 20
    x_data = design.get_samples(num_init_data_points)
    print(x_data)
    print(space)


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


if __name__ == "__main__":
    # python src/models/emulation.py
    #example_animation()
    # example_plot()
    space()
