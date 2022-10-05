"""Holland Hurricane 2008."""
import numpy as np
from scipy.optimize import curve_fit
from typing import Callable, Union, Tuple
from typeguard import typechecked
from src.data_loading.ibtracs import kat_stats


@np.vectorize
def hx(radius: float, rmax: float, xn: float, rn: float) -> float:
    """
    hx

    Args:
        radius (float): Radius [m].
        rmax (float): Radius maximum [m].
        xn (float): x value at radius_n [dimensionless].
        rn (float): radius_n [m].

    Returns:
        float: x value at radius [dimensionless].
    """
    if radius <= rmax:
        x_out = 0.5
    else:
        x_out = 0.5 + (radius - rmax) * (xn - 0.5) / (rn - rmax)
    return x_out


@np.vectorize
def sx(radius: float, rmax: float, xn: float, rn: float) -> float:
    """
    Simon's version

    Args:
        radius (float): Radius [m].
        rmax (float): Radius maximum [m].
        xn (float): x value at radius_n [dimensionless].
        rn (float): radius_n [m].

    Returns:
        float: x value at radius [dimensionless].
    """
    if radius <= rmax:
        x_out = 0.5
    else:
        x_out = xn
    return x_out


def h08(
    radius: float,
    rmax: float,
    vmax: float,
    pc: float,
    pn: float,
    r64: float,
    xn: float,
    density: float = 1.15,
) -> float:
    """
    h08.

    Args:
        radius (float): Radius [m].
        rmax (float): Radius of maximum wind [m].
        vmax (float): Velocity maximum [m s**-1].
        pc (float): Pressure central [Pa].
        pn (float): Neutral pressure [Pa].
        r64 (float): Radius of 64 knot wind [m].
        xn (float): x value at 64 knot wind.
        density (float, optional): Air density. Defaults to 1.15 [kg m**-3].

    Returns:
        float: velocity [m s**-1]
    """
    b = vmax**2 * np.e * density / (pn - pc)
    x = sx(radius, rmax, xn, r64)
    return vmax * ((rmax / radius) ** b * np.exp(1 - (rmax / radius) ** b)) ** x


@typechecked
def h08_fitfunc(
    rmax: float, vmax: float, pc: float, pn: float, r64: float, density: float = 1.04
) -> Callable:
    """
    Holland Hurricane Model fit function.

    Args:
        rmax (float): Radius of velocity maximum [m].
        vmax (float): Velocity maximum [m s**-1].
        pc (float): Central pressure [Pa].
        pn (float): Neutral pressure [Pa].
        r64 (float): Radius of 64 knot wind.
        density (float, optional): Air density [kg m**-3]. Defaults to 1.04.

    Returns:
        Callable: fit_func(radius, xn)
    """

    def v_func(radius: float, xn: float) -> float:
        return h08(radius, rmax, vmax, pc, pn, r64, xn, density=density)

    return v_func


@typechecked
def fit_h08(
    rmax: float,
    vmax: float,
    pc: float,
    pn: float,
    r64: float,
    distances: Union[list, np.ndarray],
    speeds: Union[list, np.ndarray],
    density: float = 1.04,
) -> Tuple[Callable, Callable, float]:
    """
    Holland Hurricane 2008.

    Args:
        rmax (float): Radius of velocity maximum [m s**-1].
        vmax (float): Velocity maximum [m s**-1].
        pc (float): Central pressure [Pa].
        pn (float): Neutral pressure [Pa].
        r64 (float): Radius of 64 knot wind [m].
        distances (Union[list, np.ndarray]): Distances [m].
        speeds (Union[list, np.ndarray]): Speeds [m s**-1].
        density (float, optional): Surface air density [kg]. Defaults to 1.04.

    Returns:
        Tuple[Callable, Callable, float]: v_func, p_func, xn
    """

    h08v = h08_fitfunc(rmax, vmax, pc, pn, r64, density=density)
    popt, _ = curve_fit(h08v, distances, speeds, p0=[1])

    def vel_f(xn) -> Callable:
        def vel(radius):
            return h08v(radius, xn)

        return vel

    def pres_f(xn) -> Callable:
        def pres(radius):
            b = vmax**2 * np.e * density / (pn - pc)
            return pc + (pn - pc) * np.exp(-((rmax / radius) ** b))

        return pres

    return vel_f(popt), pres_f(popt), float(popt)


if __name__ == "__main__":
    # python src/models/h08.py
    print(kat_stats())
