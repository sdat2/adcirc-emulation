"""Lines."""
from typing import Optional
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import matplotlib.path as mpath
from sithom.plot import plot_defaults
from sithom.misc import in_notebook
from src.constants import FIGURE_PATH


def colorline(
    x: np.ndarray,
    y: np.ndarray,
    z: Optional[np.ndarray] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
    label: Optional[str] = None,
    cmap: any =plt.get_cmap("copper"),
    norm: any =plt.Normalize(0.0, 1.0),
    linewidth: float = 3,
    alpha: float = 1.0,
) -> mcoll.LineCollection:
    """
    Make lines with variable colors to plot.

    https://stackoverflow.com/a/25941474

    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width

    Args:
        x (np.ndarray): The x array.
        y (np.ndarray): The y array.
        z (Optional[np.ndarray], optional): The z array. Defaults to None.
        cmap (any, optional): The colormap. Defaults to plt.get_cmap("copper").
        norm (any, optional): The colormap norm. Defaults to plt.Normalize(0.0, 1.0).
        linewidth (float, optional): The line width. Defaults to 3.
        alpha (float, optional): The line transparency. Defaults to 1.0.

    Returns:
        mcoll.LineCollection: Lines to plot.
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(
        segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha
    )
    if ax is None:
        ax = plt.gca()
    im = ax.add_collection(lc)
    if label is None:
        kwargs = {}
    else:
        kwargs = {"label": label}
    plt.colorbar(im, **kwargs, ax=ax)

    return lc


def make_segments(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    https://stackoverflow.com/a/25941474

    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array.

    Args:
        x (np.ndarray): x array.
        y (np.ndarray): y array.

    Returns:
        np.ndarray: segments.
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def test_colorline() -> None:
    """
    https://stackoverflow.com/a/25941474
    """
    N = 10
    np.random.seed(101)
    x = np.random.rand(N)
    y = np.random.rand(N)
    _, _ = plt.subplots()
    path = mpath.Path(np.column_stack([x, y]))
    verts = path.interpolated(steps=3).vertices
    x, y = verts[:, 0], verts[:, 1]
    z = np.linspace(0, 1, len(x))
    colorline(x, y, z, cmap=plt.get_cmap("jet"), linewidth=2, label="example")

    if in_notebook():
        plt.show()
    else:
        plt.savefig(os.path.join(FIGURE_PATH, "example_line_plot.png"))
        plt.clf()

    _, axs = plt.subplots(2, 1)
    colorline(
        x, y, z, cmap=plt.get_cmap("jet"), ax=axs[0], linewidth=2, label="example1"
    )
    colorline(
        x, y, z, cmap=plt.get_cmap("jet"), ax=axs[1], linewidth=2, label="example2"
    )

    if in_notebook():
        plt.show()
    else:
        plt.savefig(os.path.join(FIGURE_PATH, "example_line_multi_plot.png"))
        plt.clf()


if __name__ == "__main__":
    # python src/plot/lines.py
    plot_defaults()
    test_colorline()
