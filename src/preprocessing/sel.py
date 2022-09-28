"""Functions for selecting xarray parts."""
from typing import Union
import xarray as xr
from sithom.xr import mon_increase
from src.constants import NO_BBOX, MID_KATRINA_TIME


def mid_katrina(
    xr_obj: Union[xr.DataArray, xr.Dataset]
) -> Union[xr.DataArray, xr.Dataset]:
    """
    Mid Katrina.

    Args:
        xr_obj (Union[xr.DataArray, xr.Dataset]): Full dataset. Assume axes called "longitude", "latitude".

    Returns:
        Union[xr.DataArray, xr.Dataset]: zoomed in version midway through.

    Examples::
        >>> import xarray as xr
        >>> from src.constants import KATRINA_ERA5_NC
        >>> from src.preprocessing.sel import mid_katrina
        >>> da = xr.open_dataset(KATRINA_ERA5_NC)
        >>> sel_da = mid_katrina(da)
        >>> sel_da.tp.values.shape
        (10, 23)
    """
    if "longitude" in xr_obj:
        lons = NO_BBOX.lon
        lats = NO_BBOX.lat
        return mon_increase(xr_obj).sel(
            longitude=slice(lons[0], lons[1]),
            latitude=slice(lats[0], lats[1]),
            time=MID_KATRINA_TIME,
        )
    else:
        return xr_obj.sel(time=MID_KATRINA_TIME,)
