"""Functions for selecting xarray parts."""
from constants import ZOOMED_IN_LATS, ZOOMED_IN_LONS, MID_KATRINA_TIME
from typing import Union
import xarray as xr


def mid_katrina(
    xr_obj: Union[xr.DataArray, xr.Dataset]
) -> Union[xr.DataArray, xr.Dataset]:
    """
    Mid Katrina.

    Args:
        xr_obj (Union[xr.DataArray, xr.Dataset]): Full dataset.

    Returns:
        Union[xr.DataArray, xr.Dataset]: zoomed in version.
    """
    lons = ZOOMED_IN_LONS
    lats = ZOOMED_IN_LATS
    return xr_obj.sel(
        longitude=slice(lons[0], lons[1]),
        latitude=slice(lats[1], lats[0]),
        time=MID_KATRINA_TIME,
    )
