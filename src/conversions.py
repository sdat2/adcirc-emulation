"""Unit conversions for floats."""
import numpy as np
from sithom.place import Point
from src.constants import UREG, RADIUS_EARTH


def knots_to_ms(knots_input: float) -> float:
    """
    Knots to meter/second.

    Args:
        knots_input (float): Knots.

    Returns:
        float: meter per second output.

    Example::
        >>> from src.conversions import knots_to_ms
        >>> knots_to_ms(1)
        0.5144444444444445
    """
    return (knots_input * UREG.knot).to("metre/second").magnitude


def nmile_to_m(nmile_input: float) -> float:
    """
    Nautical Miles to meters.

    Args:
        nmile_input (float): Nautical Mile input.

    Returns:
        float: Meters output.

    Example::
        >>> from src.conversions import nmile_to_m
        >>> nmile_to_m(1.0)
        1852.0
    """
    return (nmile_input * UREG.nautical_mile).to("meter").magnitude


def distance_between_points(pt1: Point, pt2: Point) -> float:
    """
    Distance between points.

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


def millibar_to_pascal(millibar_input: float) -> float:
    """
    Millibar to pascals.

    Args:
        millibar_input (float): input in millibar.

    Returns:
        float: output in pascal.

    Example::
        >>> from src.conversions import millibar_to_pascal
        >>> millibar_to_pascal(1.0)
        100.0

    """
    return (millibar_input * UREG.millibar).to("pascal").magnitude


if __name__ == "__main__":
    # python src/conversions.py
    # print((1.0 * UREG.Radius_).to("meter").magnitude)
    print((1.0 * UREG.millibar).to("pascal").magnitude)
    # print("Run")
