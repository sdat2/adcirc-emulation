"""Functions for selecting xarray parts."""
from typing import Union, Optional, Tuple
import numpy as np
import xarray as xr
from sithom.xr import mon_increase
from sithom.place import BoundingBox
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
        return xr_obj.sel(
            time=MID_KATRINA_TIME,
        )


def trim_tri(
    x: np.ndarray,
    y: np.ndarray,
    tri: np.ndarray,
    bbox: BoundingBox,
    z: Optional[np.ndarray] = None,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    """
    Trim triangular mesh to x and y points within an area.

    Args:
        x (np.ndarray): _description_
        y (np.ndarray): _description_
        tri (np.ndarray): _description_
        bbox (BoundingBox): _description_
        z (Optional[np.ndarray], optional): z parameter. Defaults to None.

    Returns:
        _type_: _description_
    """

    @np.vectorize
    def in_bbox(xi: float, yi: float) -> bool:
        return (
            xi > bbox.lon[0]
            and xi < bbox.lon[1]
            and yi > bbox.lat[0]
            and yi < bbox.lat[1]
        )

    tindices = in_bbox(x, y)
    indices = np.where(tindices)[0]
    new_indices = np.where(indices)[0]
    neg_indices = np.where(~tindices)[0]
    tri_list = tri.tolist()
    new_tri_list = []
    for el in tri_list:
        if np.any([x in neg_indices for x in el]):
            continue
        else:
            new_tri_list.append(el)

    tri_new = np.array(new_tri_list)
    tri_new = np.select(
        [tri_new == x for x in indices.tolist()], new_indices.tolist(), tri_new
    )
    if z is None:
        return x[indices], y[indices], tri_new
    else:
        return x[indices], y[indices], tri_new, z[indices]
