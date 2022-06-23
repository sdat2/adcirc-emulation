"""IBTrACS data loading script."""
from typing import List, Tuple
import numpy as np
import xarray as xr
from sithom.time import timeit
from src.constants import IBTRACS_NC


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


@timeit
def filter_function(
    xr_obj: xr.Dataset,
    filter: List[Tuple[str, List[str]]] = [("basin", [b"NA"]), ("nature", [b"TS"]),],
) -> xr.Dataset:
    """
    Filter function for IBTrACS.

    Args:
        xr_obj (xr.DataArray): Input ibtracs datarray.
        filter (List[Tuple[str,List[str]]], optional): Filters to apply.
            Defaults to [("basin", [b"NA"]), ("nature", [b"SS", b"TS"])].

    Returns:
        xr.Dataset: Filtered dataarray.
    """
    storm_list = None
    for filter_part in filter:
        # print(filter_part)
        storm_list_part = None
        for value in filter_part[1]:
            truth_array = xr_obj[filter_part[0]] == value
            # print(truth_array.values.shape)
            compressed_array = np.any(truth_array, axis=1)
            # print(compressed_array.shape)
            storm_list_temp = xr_obj.storm.values[compressed_array]
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
    return xr_obj.sel(storm=storm_list)


def na_tcs() -> xr.Dataset:
    """
    North Atlantic Tropical Cyclones in IBTrACS.

    Returns:
        xr.Dataset: Filtered IBTrACS dataset.
    """
    return filter_function(xr.open_dataset(IBTRACS_NC))


if __name__ == "__main__":
    # python src/data_loading/ibtracs.py
    print(na_tcs())
