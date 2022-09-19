"""Unit conversions for floats."""
from src.constants import UREG


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


if __name__ == "__main__":
    # python src/conversions.py
    print((1.0 * UREG.nautical_mile).to("meter").magnitude)
