"""IBTrACS data loading script."""
from typing import Optional, List, Tuple
import numpy as np
import xarray as xr
from sithom.time import timeit
from src.constants import IBTRACS_NC, GOM_BBOX, NO_BBOX


def _union(lst1: list, lst2: list) -> list:
    """
    Union of lists.
    """
    return list(set(lst1) | set(lst2))


def _intersection(lst1: list, lst2: list) -> list:
    """
    Intersection of lists.
    """
    return list(set(lst1).intersection(set(lst2)))


# @timeit
def filter_by_labels(
    ds: xr.Dataset,
    filter: List[Tuple[str, List[str]]] = [
        ("basin", [b"NA"]),
        ("subbasin", [b"GM"]),
        ("nature", [b"TS"]),
        ("usa_record", [b"L"]),
    ],
) -> xr.Dataset:
    """
    Filter by labels for IBTrACS.

    Args:
        ds (xr.DataArray): Input ibtracs datarray.
        filter (List[Tuple[str,List[str]]], optional): Filters to apply.
            Defaults to [("basin", [b"NA"]), ("nature", [b"SS", b"TS"])].

    Returns:
        xr.Dataset: Filtered dataset.

    TODO: Relies to much on the shape of IBTRACS.

    Example of using it to filter for North Atlantic TCs::
        >>> import xarray as xr
        >>> from src.constants import IBTRACS_NC
        >>> from src.data_loading.ibtracs import filter_by_labels
        >>> ibts_ds = xr.open_dataset(IBTRACS_NC)
        >>> diff_ds = filter_by_labels(ibts_ds)
        >>> natcs_ds = filter_by_labels(ibts_ds, filter=[("basin", [b"NA"]), ("nature", [b"SS", b"TS"])])
    """
    storm_list = None
    print()
    for filter_part in filter:
        # print(filter_part)
        storm_list_part = None
        for value in filter_part[1]:
            truth_array = ds[filter_part[0]] == value
            # print(truth_array.values.shape)
            if len(truth_array.shape) != 1:
                compressed_array = np.any(truth_array, axis=1)
            else:
                compressed_array = truth_array
            # print(compressed_array.shape)
            storm_list_temp = ds.storm.values[compressed_array]
            if storm_list_part is None:
                storm_list_part = storm_list_temp
            else:
                storm_list_part = _union(storm_list_temp, storm_list_part)
            # print(len(storm_list_part))
        if storm_list is None:
            storm_list = storm_list_part
        else:
            storm_list = _intersection(storm_list_part, storm_list)
    # print(len(storm_list))
    print(len(storm_list))
    return ds.sel(storm=storm_list)


def _track_in_bbox(lons: np.ndarray, lats: np.ndarray, bbox: List[float]) -> bool:
    """
    Test if track intersects with bounding box.

    Args:
        lons (np.ndarray): Longitude array.
        lats (np.ndarray): Latitude array.
        bbox (List[float]): bounding box.

    Returns:
        bool: _description_

    Example of tracking:
        >>> from src.data_loading.ibtracs import _track_in_bbox
        >>> _track_in_bbox(np.array([0, 2]), np.array([0, 1]), [0.5, -0.5, -0.5, 0.5])
            True
        >>> _track_in_bbox(np.array([1, 2]), np.array([1, 1]), [0.5, -0.5, -0.5, 0.5])
            False
    """
    if np.any([x < bbox[0] and x > bbox[2] for x in lats]) and np.any(
        [x < bbox[3] and x > bbox[1] for x in lons]
    ):
        return True
    else:
        return False


# @np.ndarray
def _point_in_bbox(lon: float, lat: float, bbox: List[float]) -> bool:
    """
    Test if track intersects with bounding box.

    Args:
        lon (float): Longitude.
        lat (float): Latitude.
        bbox (List[float]): bounding box.

    Returns:
        bool: _description_

    Example of tracking:
        >>> from src.data_loading.ibtracs import _point_in_bbox
        >>> _point_in_bbox(0.0, 0.0, [0.5, -0.5, -0.5, 0.5])
            True
        >>> _point_in_bbox(1.0, 1.0, [0.5, -0.5, -0.5, 0.5])
            False
    """
    if lat < bbox[0] and lat > bbox[2] and lon < bbox[3] and lon > bbox[1]:
        return True
    else:
        return False


@timeit
def filter_by_bbox(ds: xr.Dataset, bbox: Optional[List[float]] = None) -> xr.Dataset:
    """
    Filter ibtracs dataset by bbox (ECMWF CDS style).

    Args:
        ds (xr.Dataset): _description_
        bbox (Optional[List[float]], optional): ECMWF style bbox. Defaults to None.

    Returns:
        xr.Dataset: xarray dataset.
    """
    if bbox is not None:
        storm_list = []
        print(bbox)
        for storm in range(ds.storm.shape[0]):
            if _track_in_bbox(
                ds.isel(storm=storm)["lon"].values,
                ds.isel(storm=storm)["lat"].values,
                bbox,
            ):
                storm_list.append(storm)
        ds = ds.isel(storm=storm_list)
    return ds


def na_tcs() -> xr.Dataset:
    """
    North Atlantic Tropical Cyclones in IBTrACS.

    Returns:
        xr.Dataset: Filtered IBTrACS dataset.
    """
    return filter_by_labels(xr.open_dataset(IBTRACS_NC))


def gom_tcs() -> xr.Dataset:
    """
    Gulf of Mexico Tropical Cyclones in IBTrACS.

    Returns:
        xr.Dataset: Filtered IBTrACS dataset
    """
    return filter_by_bbox(na_tcs(), bbox=GOM_BBOX.ecmwf())


def katrina() -> xr.Dataset:
    """
    Get Katrina.

    Returns:
        xr.Dataset: KATRINA ENTRY
    """
    return filter_by_labels(na_tcs(), filter=[("name", [b"KATRINA"])])


def no_tcs() -> xr.Dataset:
    """
    New Orleans Tropical Cyclones.

    Returns:
        xr.Dataset: xarray dataset.
    """
    return filter_by_bbox(na_tcs(), bbox=NO_BBOX.ecmwf())


if __name__ == "__main__":
    # python src/data_loading/ibtracs.py
    # print(na_tcs())
    # print(gom_tcs())
    print(katrina())
