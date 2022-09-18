"""Plot hurricane generation."""
import os
import matplotlib.pyplot as plt
from sithom.misc import in_notebook
from sithom.plot import plot_defaults
from src.models.generation import HollandTropicalCyclone
from src.plot.map import map_axes
from src.constants import NEW_ORLEANS, FIGURE_PATH


def different_trajectories() -> None:
    """
    Plot different trajectories for the idealised hurricane models.
    """
    #    plot_katrina_windfield_example(model=key)
    plot_defaults()
    ax = map_axes()

    for angle in [-80, -60, -45, -30, -15, 0, 15, 30, 45, 60, 80]:
        htc = HollandTropicalCyclone(NEW_ORLEANS, angle, 30, 70, 3000, 100)
        traj = htc.trajectory()
        plt.plot(traj[:, 0], traj[:, 1])
        plt.scatter(htc.point.lon, htc.point.lat)

    ylabel = r"Latitude [$^{\circ}$N]"
    xlabel = r"Longitude [$^{\circ}$E]"

    ax.set(xlabel=xlabel, ylabel=ylabel)
    ylocs = [20, 24, 28, 32, 36]
    xlocs = [-97, -93, -89, -85, -81]
    ax.gridlines(draw_labels=True, ylocs=ylocs, xlocs=xlocs)

    if in_notebook():
        plt.show()
    else:
        plt.savefig(os.path.join(FIGURE_PATH, "example_trajectories.png"))
        plt.clf()


if __name__ == "__main__":
    different_trajectories()
    # python src/plot/generation.py
