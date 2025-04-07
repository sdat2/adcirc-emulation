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

    # TODO: at the moment, this breaks for large meshes. Need to fix.

    Args:
        x (np.ndarray): longitude [degrees East].
        y (np.ndarray): latitude [degrees North].
        tri (np.ndarray): triangular mesh.
        bbox (BoundingBox): bounding box to trim by.
        z (Optional[np.ndarray], optional): z parameter. Defaults to None.

    Returns:
        Union[
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
            Tuple[np.ndarray, np.ndarray, np.ndarray],
             ]: trimmed x, y, tri, (and z).
    """
    print("num_old_indices", len(x))

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
    print("num_new_indices", len(new_indices))
    print("Trimming mesh...")

    for el in tri_list:
        if np.any([x in neg_indices for x in el]):
            continue
        else:
            new_tri_list.append(el)

    tri_new = np.array(new_tri_list)
    # should there be an off by one error here? I think not, graphs look fine.
    tri_new = np.select(
        [tri_new == x for x in indices.tolist()], new_indices.tolist(), tri_new
    )
    if z is None:
        return x[indices], y[indices], tri_new
    elif len(z.shape) == 1:
        return x[indices], y[indices], tri_new, z[indices]
    elif len(z.shape) == 2:
        return x[indices], y[indices], tri_new, z[:, indices]
    else:
        return None


def test_trim(plot: bool = False):
    """Test trim_tri."""
    x = np.array([0, 1, 2, 2.5, 2, 2.2])  # knock out  # knock out
    new_x = np.array([1, 2, 2.5, 2.2])
    y = np.array([0, 1.2, 1, 2.5, 4.5, 1.2])
    new_y = np.array([1.2, 1, 2.5, 1.2])
    tri = np.array(
        [[0, 1, 2], [1, 2, 3], [2, 3, 4], [0, 2, 4], [0, 1, 4], [1, 2, 5], [2, 3, 5]]
    )
    new_tri = np.array([[0, 1, 2], [0, 1, 3], [1, 2, 3]])  # relabel and knock out
    bbox = BoundingBox(lon=[0.5, 2.6], lat=[0.5, 2.6])
    test_x, test_y, test_tri = trim_tri(x, y, tri, bbox)
    # print(test_x, test_y, test_tri)
    assert np.allclose(test_x, new_x)
    assert np.allclose(test_y, new_y)
    assert np.allclose(test_tri, new_tri)
    if plot:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        plt.triplot(x, y, tri, color="grey", label="old")
        rect = patches.Rectangle(
            (bbox.lon[0], bbox.lat[0]),
            bbox.lon[1] - bbox.lon[0],  # width
            bbox.lat[1] - bbox.lat[0],  # height
            color="red",
            rotation_point="xy",
            facecolor="none",
            fill=False,
        )
        plt.gca().add_patch(rect)
        plt.triplot(test_x, test_y, test_tri, color="green", label="new")
        plt.legend()
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.show()


if __name__ == "__main__":
    # python src/preprocessing/sel.py
    test_trim(plot=True)
