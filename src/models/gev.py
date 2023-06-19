"""
GEV.

Initially based on stackoverflow answer:
https://stackoverflow.com/a/52460086
"""
import os
from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genextreme as gev
from pyextremes import get_extremes, get_return_periods
from sithom.time import timeit
from sithom.plot import plot_defaults, label_subplots
from src.constants import DATA_PATH, FIGURE_PATH


def fit_gev(rvs: float) -> Tuple[float, float, float]:
    """Fit GEV."""
    shape, loc, scale = gev.fit(rvs)
    return shape, loc, scale


def generate_samples(shape: float, loc: float, scale: float, size: int) -> np.array:
    """Generate samples."""
    return gev.rvs(shape, loc, scale, size=size)


@timeit
def sample_effect_exp(
    num: int = 50, shape: int = -1, loc: int = 1, size: int = 1
) -> Tuple[np.array, np.array, np.array, np.array]:
    """Sample effect experiment.

    Args:
        num (int, optional): Number of samples. Defaults to 50.
        shape (int, optional): Shape parameter. Defaults to -1.
        loc (int, optional): Location parameter. Defaults to 1.
        size (int, optional): Size parameter. Defaults to 1.
    """
    # O(num * samples * UB/2)
    samps = np.linspace(10, 1000, num=num, dtype=int)
    sh_ll, l_ll, sc_ll = [], [], []
    for j in samps:
        sh_l, l_l, sc_l = [], [], []
        for _ in range(100):
            sh, l, sc = fit_gev(generate_samples(shape, loc, size, j))
            sh_l.append(sh)
            l_l.append(l)
            sc_l.append(sc)

        sh_ll.append(sh_l)
        l_ll.append(l_l)
        sc_ll.append(sc_l)
        print("xi", np.mean(sh_l), np.std(sh_l))
        print("mu", np.mean(l_l), np.std(l_l))
        print("sigma", np.mean(sc_l), np.std(sc_l))
    return np.array(samps), np.array(sh_ll), np.array(l_ll), np.array(sc_ll)


def gen_isf(q: float, shape: float, loc: float, scale: float) -> float:
    return gev.isf(q, shape, loc, scale)


def gev_f(x: float, shape: float, loc: float, scale: float) -> float:
    return gev.pdf(x, shape, loc, scale)


def cost_function(
    x: float,
    alpha=1000,  # cost function
    real_max=2,
) -> callable:
    """Cost function.

    Args:
        x (float): Input data.
        alpha (int, optional): Cost function parameter. Defaults to 1000.
        real_max (int, optional): Cost function parameter. Defaults to 2.

    Returns:
        callable: Cost function.
    """

    def cf(p):
        shp = p[0]
        loc = p[1]
        scale = p[2]
        return (
            np.sum(gev.pdf(x, shp, loc, scale))
            + alpha * (scale / shp + loc - real_max) ** 2
        )

    return cf


def minimize_wrt_gev(
    x: float,
    shape: float,  # initial guess
    loc: float,
    scale: float,
    alpha=1000,  # cost function
    real_max=2,
) -> callable:
    """Minimize wrt GEV.

    Args:
        x (float): Input data.
        shape (float): Initial guess.
        loc (float): Initial guess.
        scale (float): Initial guess.
        alpha (int, optional): Cost function parameter. Defaults to 1000.
        real_max (int, optional): Cost function parameter. Defaults to 2.

    Returns:
        callable: Minimized function."""
    cf = cost_function(x, alpha=alpha, real_max=real_max)
    import scipy.optimize as opt

    big = 10

    return opt.minimize(
        cf,
        [shape, loc, scale],
        bounds=[(0, big), (-big, big), (0, big)],
        # method="Nelder-Mead",
    )


def min_test() -> None:
    p = minimize_wrt_gev(gev.rvs(1, 1, 1, size=1000), 0.9, 0.9, 0.9)
    print(p)


@timeit
def plot_sample_exp(
    samp: np.ndarray,
    shp: np.ndarray,
    loc: np.ndarray,
    scale: np.ndarray,
    oshp: float = -1,
    oloc: float = 1,
    oscale: float = 1,
    figure_path: str = os.path.join(FIGURE_PATH, "gev_exp"),
) -> None:
    """Plot sample effect experiments on different graphs.

    What difference does the number of samples make on the convergence of the GEV parameters?

    Args:
        samp (np.ndarray): Samples.
        shp (np.ndarray): Shapes.
        loc (np.ndarray): Locations.
        scale (np.ndarray): Scales.
        oshp (float, optional): Original shape. Defaults to -1.
        oloc (float, optional): Original location. Defaults to 1.
        oscale (float, optional): Original scale. Defaults to 1.

    Returns:
        None: None.
    """
    os.makedirs(figure_path, exist_ok=True)
    plot_defaults()
    print("samp.shape", "shp.shape", "loc.shape", "scale.shape")
    print(samp.shape, shp.shape, loc.shape, scale.shape)

    def tri_setup() -> any:
        fig, axs = plt.subplots(3, 1, sharex=True, sharey=True)
        plt.xlim(10, 1000)
        plt.ylim(-1, 1)
        axs[2].set_xlabel("Number of samples")
        axs[0].set_ylabel("$\Delta$ Shape")
        axs[1].set_ylabel("$\Delta$ Location")
        axs[2].set_ylabel("$\Delta$ Scale")
        label_subplots(axs)
        return axs

    axs = tri_setup()
    axs[0].semilogx(
        samp, np.sort(shp, axis=1) - oshp, linewidth=0.3, alpha=0.5, color="red"
    )
    axs[1].semilogx(
        samp, np.sort(loc, axis=1) - oloc, linewidth=0.3, alpha=0.5, color="blue"
    )
    axs[2].semilogx(
        samp, np.sort(scale, axis=1) - oscale, linewidth=0.3, alpha=0.5, color="green"
    )
    plt.savefig(os.path.join(figure_path, "gev_exp_sort.png"))
    plt.clf()

    axs = tri_setup()
    mn = np.mean(shp, axis=1) - oshp
    std = np.std(shp, axis=1)
    axs[0].semilogx(samp, mn, linewidth=1, alpha=0.5, color="red")
    axs[0].fill_between(samp, mn - std, mn + std, alpha=0.5, color="red")
    mn = np.mean(loc, axis=1) - oloc
    std = np.std(loc, axis=1)
    axs[1].semilogx(samp, mn, linewidth=1, alpha=0.5, color="blue")
    axs[1].fill_between(samp, mn - std, mn + std, alpha=0.5, color="blue")
    mn = np.mean(scale, axis=1) - oscale
    std = np.std(scale, axis=1)
    axs[2].semilogx(samp, mn, linewidth=1, alpha=0.5, color="green")
    axs[2].fill_between(samp, mn - std, mn + std, alpha=0.5, color="green")
    plt.savefig(os.path.join(figure_path, "gev_exp_mnstd.png"))
    plt.clf()

    def heights(exceedance_probability: 0.01):
        heights = np.zeros(np.shape(shp))
        for i in range(np.shape(shp)[0]):
            for j in range(np.shape(shp)[1]):
                heights[i, j] = gev.isf(
                    exceedance_probability, shp[i, j], loc[i, j], scale[i, j]
                )
        # return gen_isf(x, shp, loc, scale)
        # np.sort(shp, axis=1)
        return np.sort(heights, axis=1)

    heights_100 = heights(1 / 100)
    heights_1_000 = heights(1 / 1_000)
    heights_100_000 = heights(1 / 100_000)

    def aep_setup() -> any:
        fig, axs = plt.subplots(3, 1, sharex=True, sharey=False)
        plt.xlim(10, 1000)
        axs[0].set_ylim(0, heights_100[10, 98])
        axs[1].set_ylim(0, heights_1_000[10, 99])
        axs[2].set_ylim(0, heights_100_000[10, 99])
        axs[2].set_xlabel("Number of samples")
        axs[0].set_ylabel("100 years")
        axs[1].set_ylabel("1,000 year")
        axs[2].set_ylabel("100,000 year")
        label_subplots(axs)
        return axs

    axs = aep_setup()

    oheight_100 = gen_isf(1 / 100, oshp, oloc, oscale)
    oheight_1_000 = gen_isf(1 / 1_000, oshp, oloc, oscale)
    oheight_100_000 = gen_isf(1 / 100_000, oshp, oloc, oscale)

    axs[0].semilogx(samp, heights_100, linewidth=1, alpha=0.5, color="red")
    axs[0].plot(
        samp,
        np.ones(np.shape(samp)) * oheight_100,
        linewidth=1,
        alpha=0.5,
        color="black",
    )
    axs[1].semilogx(samp, heights_1_000, linewidth=1, alpha=0.5, color="blue")
    axs[1].plot(
        samp,
        np.ones(np.shape(samp)) * oheight_1_000,
        linewidth=1,
        alpha=0.5,
        color="black",
    )
    axs[2].semilogx(samp, heights_100_000, linewidth=1, alpha=0.5, color="green")
    axs[2].plot(
        samp,
        np.ones(np.shape(samp)) * oheight_100_000,
        linewidth=1,
        alpha=0.5,
        color="black",
    )

    plt.savefig(os.path.join(figure_path, "gev_exp_heights.png"))
    plt.clf()

    # fill between plots
    axs = aep_setup()
    mn = np.mean(heights_100, axis=1)
    std = np.std(heights_100, axis=1)
    axs[0].semilogx(samp, mn, linewidth=1, alpha=0.5, color="red")
    axs[0].semilogx(samp, heights_100[:, 49], "--", linewidth=1, alpha=0.5, color="red")
    axs[0].plot(
        samp,
        np.ones(np.shape(samp)) * oheight_100,
        linewidth=1,
        alpha=0.5,
        color="black",
    )
    axs[0].fill_between(samp, mn - std, mn + std, alpha=0.5, color="red")
    mn = np.mean(heights_1_000, axis=1)
    std = np.std(heights_1_000, axis=1)
    axs[1].semilogx(samp, mn, linewidth=1, alpha=0.5, color="blue")
    axs[1].semilogx(
        samp, heights_1_000[:, 49], "--", linewidth=1, alpha=0.5, color="blue"
    )
    axs[1].plot(
        samp,
        np.ones(np.shape(samp)) * oheight_1_000,
        linewidth=1,
        alpha=0.5,
        color="black",
    )
    axs[1].fill_between(samp, mn - std, mn + std, alpha=0.5, color="blue")
    mn = np.mean(heights_100_000, axis=1)
    std = np.std(heights_100_000, axis=1)
    axs[2].semilogx(samp, mn, linewidth=1, alpha=0.5, color="green")
    axs[2].semilogx(
        samp, heights_100_000[:, 49], "--", linewidth=1, alpha=0.5, color="green"
    )
    axs[2].plot(
        samp,
        np.ones(np.shape(samp)) * oheight_100_000,
        linewidth=1,
        alpha=0.5,
        color="black",
    )
    axs[2].fill_between(samp, mn - std, mn + std, alpha=0.5, color="green")
    plt.savefig(os.path.join(figure_path, "gev_exp_heights_mnstd.png"))
    plt.clf()


def exp_and_plot(
    data_calculated=False, shape: float = -1, location: float = 1, scale: float = 1
) -> None:
    """Generate samples and plot.

    Args:
        data_calculated (bool, optional): Whether data has already been calculated. Defaults to False.
        shape (float, optional): Shape parameter. Defaults to -1.
        location (float, optional): Location parameter. Defaults to 1.
        scale (float, optional): Scale parameter. Defaults to 1.

    """
    path = f"gev_exp_{str(shape)}_{str(location)}_{str(scale)}"
    data_dir = os.path.join(DATA_PATH, path)
    figure_dir = os.path.join(FIGURE_PATH, path)

    for dir in [data_dir, figure_dir]:
        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)

    if not data_calculated:
        samp, shp, loc, size = sample_effect_exp(shape=shape, loc=location, size=scale)
        # save numpy arrays
        np.save(os.path.join(data_dir, "gev_samp.npy"), samp)
        np.save(os.path.join(data_dir, "gev_shp.npy"), shp)
        np.save(os.path.join(data_dir, "gev_loc.npy"), loc)
        np.save(os.path.join(data_dir, "gev_scale.npy"), size)

    else:
        # load data
        samp = np.load(os.path.join(data_dir, "gev_samp.npy"))
        shp = np.load(os.path.join(data_dir, "gev_shp.npy"))
        loc = np.load(os.path.join(data_dir, "gev_loc.npy"))
        size = np.load(os.path.join(data_dir, "gev_scale.npy"))

    # plot
    plot_sample_exp(
        samp,
        shp,
        loc,
        size,
        oshp=shape,
        oloc=location,
        oscale=scale,
        figure_path=figure_dir,
    )


def example():
    rvs: List[float] = [
        9.4,
        38.0,
        12.5,
        35.3,
        17.6,
        12.9,
        12.4,
        19.6,
        15.0,
        13.2,
        12.3,
        16.9,
        16.9,
        29.4,
        13.6,
        11.1,
        8.0,
        16.6,
        12.0,
        13.1,
        9.1,
        9.7,
        21.0,
        11.2,
        14.4,
        18.8,
        14.0,
        19.9,
        12.4,
        10.8,
        21.6,
        15.4,
        17.4,
        14.8,
        22.7,
        11.5,
        10.5,
        11.8,
        12.4,
        16.6,
        11.7,
        12.9,
        17.8,
    ]
    shape, loc, scale = fit_gev(rvs)

    print("xi", shape)
    print("mu", loc)
    print("sigma", scale)
    # minima?
    l = loc + scale / shape
    # domain for plotting (in mm)
    xx = np.linspace(l + 0.00001, l + 0.00001 + 35, num=10000)
    # probability density function
    yy = gev.pdf(xx, shape, loc, scale)
    plt.plot(xx, yy, "r", label="PDF")
    # cumulative distribution function
    yy = gev.cdf(xx, shape, loc, scale)
    plt.plot(xx, yy, "orange", label="CDF")
    # inverse survival function
    yy = gev.sf(xx, shape, loc, scale)
    plt.plot(xx, yy, "g", label="SF")
    # plot histogram of input data
    hist, bins = np.histogram(rvs, bins=12, range=(-0.5, 23.5), density=True)
    plt.bar(bins[:-1], hist, width=2, align="edge")
    plt.xlabel("Rainfall (mm)")
    plt.ylabel("Probability")
    plt.legend()

    plt.show()
    plt.semilogx(
        1 / yy,
        xx,
        label="GEV (Frechet) distribution",
    )
    print("yy", yy)
    print("xx", xx)

    # sorted random variables
    xx = np.sort(rvs)[::-1]
    # rank
    yy = (len(xx) + 1) / np.linspace(1, len(xx), num=len(xx))
    plt.plot(yy, xx, "x", label="Data points")

    plt.xlabel("1/p, Return period (years)")
    plt.ylabel("Exceedance level, Rainfall (mm)")
    plt.show()


def ok():
    rvs = []
    extremes = get_extremes(
        ts=np.asarray(rvs),
        method="BM",
        block_size="365.2425D",
    )
    return_periods = get_return_periods(
        ts=np.asarray(rvs),
        extremes=extremes,
        extremes_method="BM",
        extremes_type="high",
        block_size="365.2425D",
        return_period_size="365.2425D",
        plotting_position="weibull",
    )
    plt.show()
    # plt.plot(return_periods["return period"], return_periods["return level"])
    return_periods.sort_values("return period", ascending=False).head()


def multiple_tests():
    exp_and_plot(data_calculated=True, shape=-1, location=1, scale=1)
    exp_and_plot(data_calculated=True, shape=-0.5, location=1, scale=1)
    exp_and_plot(data_calculated=True, shape=-0.5, location=1, scale=1)
    exp_and_plot(data_calculated=True, shape=-0.25, location=1, scale=1)
    exp_and_plot(data_calculated=True, shape=-0, location=1, scale=1)
    exp_and_plot(data_calculated=True, shape=0.25, location=1, scale=1)
    exp_and_plot(data_calculated=True, shape=0.5, location=1, scale=1)
    exp_and_plot(data_calculated=True, shape=0.75, location=1, scale=1)
    exp_and_plot(data_calculated=True, shape=1, location=1, scale=1)

    # example()


if __name__ == "__main__":
    min_test()
