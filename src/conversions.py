"""Unit conversions."""
from typing import Union
import numpy as np
import xarray as xr
from sithom.place import Point
from src.constants import UREG, RADIUS_EARTH


def knots_to_ms(knots_input: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Knots to meter/second.

    Args:
        knots_input (Union[float, np.ndarray]): Knots.

    Returns:
        Union[float, np.ndarray]t: meter per second output.

    Example::
        >>> from src.conversions import knots_to_ms
        >>> knots_to_ms(1)
        0.5144444444444445
    """
    return (knots_input * UREG.knot).to("metre/second").magnitude


def nmile_to_meter(nmile_input: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Nautical Miles to meters.

    Args:
        nmile_input (Union[float, np.ndarray]): Nautical Mile input.

    Returns:
        Union[float, np.ndarray]: Meters output.

    Example::
        >>> from src.conversions import nmile_to_meter
        >>> nmile_to_meter(1.0)
        1852.0
    """
    return (nmile_input * UREG.nautical_mile).to("meter").magnitude


def meter_to_nmile(meter_input: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Meters to Nautical Miles.

    Args:
        meter_input (Union[float, np.ndarray]): meter input.

    Returns:
        Union[float, np.ndarray]: nmile output.

    Example::
        >>> from src.conversions import meter_to_nmile
        >>> meter_to_nmile(1852.0)
        1.0
    """
    return (meter_input * UREG.meter).to("nautical_mile").magnitude



def distance_between_points(pt1: Point, pt2: Point) -> float:
    """
    Distance between lon/lat points.

    NOTE: Approximates Earth as perfect sphere.

    Args:
        pt1 (Point): first lon lat point.
        pt2 (Point): second lon lat point.

    Returns:
        float: distance.

    Examples::
        >>> from sithom.place import Point
        >>> from src.conversions import distance_between_points
        >>> point1 = Point(0, 0)
        >>> point2 = Point(10, 0)
        >>> distance_a = distance_between_points(point1, point2)
        >>> distance_a
        1111950.8372419141
        >>> point3 = Point(0, 10)
        >>> distance_b = distance_between_points(point1, point3)
        >>> distance_b
        1111950.8372419141

    """
    rlat1 = np.radians(pt1.lat)
    rlat2 = np.radians(pt2.lat)
    dlat = np.radians(pt2.lat - pt1.lat)
    dlong = np.radians(pt2.lon - pt1.lon)

    alpha = np.sin(dlat / 2) * np.sin(dlat / 2) + np.cos(rlat1) * np.cos(
        rlat2
    ) * np.sin(dlong / 2) * np.sin(dlong / 2)
    radians = 2 * np.arctan2(np.sqrt(alpha), np.sqrt(1 - alpha))
    return (RADIUS_EARTH * radians).to("meter").magnitude


def distances_to_points(point: Point, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    """
    Distance to points.

    Args:
        point (Point): Central point (lon, lat).
        lons (np.ndarray): Longitude array (degrees_East).
        lats (np.ndarray): Latitude array (degrees_North).

    Returns:
        np.ndarray: Distance array (meters).


    Example:
        >>> import numpy as np
        >>> from sithom.place import Point
        >>> point_a = Point(0, 0)
        >>> lons = np.array([0, 1, 2])
        >>> lats = np.array([0, 0, 0])
        >>> distances = distances_to_points(point_a, lons, lats)
        >>> distances.tolist()
        [0.0, 111195.08372419143, 222390.16744838285]

    """
    assert lons.shape == lats.shape

    @np.vectorize
    def distance(lon: float, lat: float) -> float:
        return distance_between_points(point, Point(lon, lat))

    return distance(lons, lats)


def angle_between_points(point1: Point, point2: Point) -> float:
    """
    Angle from point2 to point1.

    Args:
        point1 (Point): target
        point2 (Point): gun

    Returns:
        float: angle in degrees.

    Example::
        >>> angle_between_points(Point(0,0), Point(0, 10))
        180.0
        >>> angle_between_points(Point(0,0), Point(0, -10))
        0.0
        >>> angle_between_points(Point(0,0), Point(10, 0))
        -90.0
        >>> angle_between_points(Point(0,0), Point(-10, 0))
        90.0
    """
    myradians = np.arctan2(
        np.radians(point1.lon - point2.lon), np.radians(point1.lat - point2.lat)
    )
    #  If you want to convert radians to degrees
    return np.degrees(myradians)


def angles_to_points(point: Point, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    """
    angles to points.

    Args:
        point (Point): point.
        lons (np.ndarray): longitude.
        lats (np.ndarray): latitude.

    Returns:
        np.ndarray: numpy array of degree angles

    Example::
        >>> point_a = Point(0, 0)
        >>> lons = np.array([0, 10, 0, -10])
        >>> lats = np.array([10, 0, -10, 0])
        >>> angles = angles_to_points(point_a, lons, lats)
        >>> angles.tolist()
        [180.0, -90.0, 0.0, 90.0]

    """

    assert lons.shape == lats.shape

    @np.vectorize
    def angle(lon: float, lat: float) -> float:
        return angle_between_points(point, Point(lon, lat))

    return angle(lons, lats)


def millibar_to_pascal(
    millibar_input: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Millibar to pascals.

    Args:
        millibar_input (Union[float, np.ndarray]): input in millibar.

    Returns:
        Union[float, np.ndarray]: output in pascal.

    Example::
        >>> from src.conversions import millibar_to_pascal
        >>> millibar_to_pascal(1.0)
        100.0

    """
    return (millibar_input * UREG.millibar).to("pascal").magnitude


def pascal_to_millibar(
    pascal_input: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Pascal to millibar.

    Args:
        pascal_input (Union[float, np.ndarray]): input in pascal.

    Returns:
        Union[float, np.ndarray]: output in millibar.

    Example::
        >>> from src.conversions import pascal_to_millibar
        >>> pascal_to_millibar(100.0)
        1.0
    """
    return (pascal_input * UREG.pascal).to("millibar").magnitude


def kelvin_to_celsius(
    kelvin_input: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Kelvin to celsius.

    Args:
        kelvin_input (Union[float, np.ndarray]Union[float, np.ndarray]): input in Kelvin.

    Returns:
        Union[float, np.ndarray]: output in degrees Celsius.

    Example::
        >>> from src.conversions import kelvin_to_celsius
        >>> kelvin_to_celsius(1.0)
        -272.15
    """
    return (kelvin_input * UREG.kelvin).to("celsius").magnitude


def celsius_to_kelvin(
    celsius_input: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Degrees Celsius to Kelvin.

    Args:
        celsius_input (Union[float, np.ndarray]): input in degrees Celsius.

    Returns:
        Union[float, np.ndarray]: output in Kelvin.

    Example::
        >>> from src.conversions import celsius_to_kelvin
        >>> celsius_to_kelvin(1.0)
        274.15
    """
    return (celsius_input * UREG.celsius).to("kelvin").magnitude


def fcor_from_lat(lat: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    F-corriolis coefficient from latitude (degrees North).

    .. math::
        :nowrap:
        \\begin{equation}
            f=2 \\Omega \\sin \\phi
        \\end{equation}

    Where phi is the latitude, and omega is the planetary
    angular frequency.

    Args:
        lat (Union[float, np.ndarray]): Latitude (degrees North).

    Returns:
        Union[float, np.ndarray]: Coriolis coefficient.

    TODO: Should there be an np.abs in this function? 7.2921e-5

    Example::
        >>> from src.conversions import fcor_from_lat
        >>> fcor_from_lat(90) / 2
        7.27220521664304e-05

    """
    return (
        2.0
        * 2.0
        * np.pi
        / ((1.0 * UREG.day).to("second").magnitude)
        * np.sin(np.radians(np.abs(lat)))
    )


def si_ify(input: xr.Dataset) -> xr.Dataset:
    """
    SI-ify

    TODO: could change to work for xr.DataArray as well.

    Args:
        input (xr.Dataset): dataset to change to SI units.

    Returns:
        xr.Dataset: Dataset with SI units instead.

    Example::
        >>> import numpy as np
        >>> import xarray as xr
        >>> from src.conversions import si_ify
        >>> wsp_knts = [[1.0, 1.0], [1.0, 1.0]]
        >>> lon = [[-99.83, -99.32], [-99.79, -99.23]]
        >>> lat = [[42.25, 42.21], [42.63, 42.59]]
        >>> ds = xr.Dataset(
        ...    data_vars=dict(
        ...        wsp=(["x", "y"], wsp_knts),
        ...        ),
        ...    coords=dict(
        ...        lon=(["x", "y"], lon),
        ...        lat=(["x", "y"], lat),
        ...        )
        ...    )
        >>> ds["wsp"].attrs["units"] = "knots"
        >>> ds = si_ify(ds)
        >>> np.all(ds.wsp.values == 0.5144444444444445)
        True
        >>> ds.wsp.attrs["units"]
        'm s**-1'

    """

    # Map different names for legacy units to those
    # in SI_DICT.
    rename_dict = {"kts": "knots", "millibar": "mb"}

    # OLD_UNIT : (NEW_UNIT, conversion_function)

    si_dict = {
        "knots": ("m s**-1", knots_to_ms),
        "nmile": ("m", nmile_to_meter),
        "mb": ("Pa", millibar_to_pascal),
    }

    for var in input:
        if "units" in input[var].attrs:
            init_unit = input[var].attrs["units"]
            if init_unit in rename_dict:
                init_unit = rename_dict[init_unit]
            if init_unit in si_dict:
                input[var][:] = si_dict[init_unit][1](input[var].values)
                input[var].attrs["units"] = si_dict[init_unit][0]

    return input


if __name__ == "__main__":
    # python src/conversions.py
    # print((1.0 * UREG.Radius_).to("meter").magnitude)
    print((1.0 * UREG.year).to("second").magnitude)
    # print("Run")
