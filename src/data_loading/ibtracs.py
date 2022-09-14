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


def filter_by_bbox(ds: xr.Dataset, bbox: Optional[List[float]] = None) -> xr.Dataset:
    """
    Filter ibtracs dataset by bbox (ECMWF CDS style).

    Takes about 10 seconds to run over Gulf of Mexico subset.

    Args:
        ds (xr.Dataset): IBTrACS dataset.
        bbox (Optional[List[float]], optional): ECMWF style bbox. Defaults to None.

    Returns:
        xr.Dataset: xarray dataset.
    """
    if bbox is not None:
        storm_list = []
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


def landings_only(ds: xr.Dataset) -> xr.Dataset:
    """
    Extract a reduced dataset based on those points at which a landing occurs.

    Args:
        ds (xr.Dataset): Individual storm.

    Returns:
        xr.Dataset: Clipped storm.
    """
    return ds.isel(date_time=ds["date_time"][(ds["usa_record"].values == b"L").ravel()])


def landing_distribution(
    ds: xr.Dataset, var: str = "usa_pres", sanitize: bool = True
) -> np.ndarray:
    """
    Landing distribution.

    Args:
        ds (xr.Dataset): IBTrACS dataset.
        var (str, optional): Variable. Defaults to "usa_pres".
        sanitize (bool, optional). Whether to remove NaNs. Defaults to True.

    Returns:
        list: List of outputs. Can include NaNs if sanitize==False.

    Example::
        >>> landing_distribution(katrina()).tolist()
        [984.0, 920.0, 928.0]
    """
    output = []
    for storm in ds["storm"].values:
        landing_ds = landings_only(ds.isel(storm=storm))
        output += landing_ds[var].values.tolist()

    if sanitize:
        output = [x for x in output if str(x) != "nan"]

    return np.array(output)


# from adcircpy.forcing.winds._parametric.holland2010 import holland_B
@np.vectorize
def holland_b(
    vmax: float,
    rmax: float,
    neutral_pressure: float,
    central_pressure: float,
    eye_latitude: float,
) -> float:
    """
    Calculate Holland 2010 B parameter.

    Args:
        vmax (float): _description_
        rmax (float): _description_
        neutral_pressure (float): _description_
        central_pressure (float): _description_
        eye_latitude (float): _description_

    Returns:
        float: Holland 2010 B parameter.
    """
    air_density = 1.15
    if neutral_pressure <= central_pressure:
        neutral_pressure = np.nan  # central_pressure + 1.0
    f = 2.0 * 7.2921e-5 * np.sin(np.radians(np.abs(eye_latitude)))
    return (vmax**2 + vmax * rmax * f * air_density * np.exp(1)) / (
        neutral_pressure - central_pressure
    )


def holland_b_usa(ds: xr.Dataset) -> np.ndarray:
    """
    Calculate Holland 2010 B parameter using US variables.

    Args:
        ds (xr.Dataset): Individual IBTRACS storm.

    Returns:
        np.ndarray: B parameters.
    """
    var_names = ["usa_wind", "usa_rmw", "usa_poci", "usa_pres", "usa_lat"]
    var_list = [ds[var].values for var in var_names]
    return holland_b(*var_list)


def holland_b_landing_distribution(ds: xr.Dataset, sanitize: bool = True) -> np.ndarray:
    """
    Calculate Holland 2010 B parameter distribution.

    Args:
        ds (xr.Dataset): _description_
        sanitize (bool, optional): _description_. Defaults to True.

    Returns:
        List[float]: Holland B parameters.

    Example::
        >>> holland_b_landing_distribution(katrina()).tolist()
        [175.00549603625592, 140.7033819399621, 143.1886980704015]
    """
    output = []
    for storm in ds["storm"].values:
        landing_ds = landings_only(ds.isel(storm=storm))
        output += holland_b_usa(landing_ds).tolist()

    if sanitize:
        output = [x for x in output if str(x) != "nan"]

    output = np.array(output)

    return output[output < 700]


def time_steps(input: xr.Dataset) -> xr.Dataset:
    """
    Add time steps to the IBTrACS dataset.

    Args:
        input (xr.Dataset): IBTrACS dataset.

    Returns:
        xr.Dataset: Input with it calculated.

    Example:
        >>> from src.data_loading.ibtracs import katrina, time_steps, na_tcs
        >>> kat_steps = time_steps(katrina())
        >>> kat_steps["time_step"].values.shape
        (1, 360)
        >>> time_steps(na_tcs())["time_step"].values.shape
        (465, 360)
    """
    times = input.time.values
    time_steps_list = [
        (times[:, i + 1] - times[:, i]) / np.timedelta64(1, "h") for i in range(359)
    ]
    time_steps_list.append(np.array([np.nan for _ in range(len(times))]))
    time_steps = np.array(time_steps_list).transpose()
    input["time_step"] = (["storm", "date_time"], time_steps)
    input["time_step"].attrs["units"] = "hours"
    return input


def prep_for_climada(input: xr.Dataset) -> xr.Dataset:
    """
    Prepare IBTrACS for being a climada input.

    Args:
        input (xr.Dataset):

    Returns:
        xr.Dataset:
    """
    rename_dict = {
        "radius_max_wind": "usa_rmw",
        "environmental_pressure": "usa_poci",
        "central_pressure": "usa_pres",
        "max_sustained_wind": "usa_wind",
    }

    time_steps(input)

    for key in rename_dict:
        input[key] = input[rename_dict[key]]

    # required = ['lat', 'lon', 'time_step', 'radius_max_wind','environmental_pressure', 'central_pressure']

    if len(input.storm.values) == 1:
        input = input.isel(storm=0)

    return input


if __name__ == "__main__":
    # python src/data_loading/ibtracs.py
    # print(na_tcs())
    # print(gom_tcs())
    print(katrina())
