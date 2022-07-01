"""Functions for selecting xarray parts."""
from typing import Union
import xarray as xr
from src.constants import NO_BBOX, MID_KATRINA_TIME


def mid_katrina(
    xr_obj: Union[xr.DataArray, xr.Dataset]
) -> Union[xr.DataArray, xr.Dataset]:
    """
    Mid Katrina.

    Args:
        xr_obj (Union[xr.DataArray, xr.Dataset]): Full dataset.

    Returns:
        Union[xr.DataArray, xr.Dataset]: zoomed in version midway through.
    """
    lons = NO_BBOX.lon
    lats = NO_BBOX.lat
    return xr_obj.sel(
        longitude=slice(lons[0], lons[1]),
        latitude=slice(lats[1], lats[0]),
        time=MID_KATRINA_TIME,
    )
