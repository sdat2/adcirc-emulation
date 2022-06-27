
from typing import Union
import xarray as xr

ZOOMED_IN_LONS = [-92, -86.5] # zoomed in around New orleans
ZOOMED_IN_LATS = [28.5, 30.8]

def mid_katrina(xr_obj: Union[xr.DataArray, xr.Dataset]) -> Union[xr.DataArray, xr.Dataset]:
    """
    Mid Katrina.

    Args:
        xr_obj (Union[xr.DataArray, xr.Dataset]): Full dataset.

    Returns:
        Union[xr.DataArray, xr.Dataset]: zoomed in version.
    """
    lons = ZOOMED_IN_LONS
    lats = ZOOMED_IN_LATS
    return xr_obj.sel(longitude=slice(lons[0], lons[1]), latitude=slice(lats[1], lats[0]),  time="2005-08-29T10:00:00")
