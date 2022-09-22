"""EMUKIT"""
import os
import numpy as np
import matplotlib.pyplot as plt
import GPy
from emukit.experimental_design.experimental_design_loop import ExperimentalDesignLoop
from emukit.model_wrappers import SimpleGaussianProcessModel
from emukit.core import ParameterSpace, ContinuousParameter
from emukit.core.loop import UserFunctionWrapper
from sithom.plot import plot_defaults
from sithom.misc import in_notebook
from src.constants import FIGURE_PATH
from emukit.core.initial_designs.latin_design import LatinDesign


def example_plot() -> None:
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


if __name__ == "__main__":
    # python src/models/emukit.py
    example_plot()
