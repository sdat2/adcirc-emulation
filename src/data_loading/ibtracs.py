"""IBTrACS data loading script."""
from typing import Callable, Optional, List, Tuple
import warnings
import numpy as np
from scipy import optimize
import xarray as xr
from src.constants import IBTRACS_NC, GOM_BBOX, NO_BBOX
from src.conversions import fcor_from_lat

# from sithom.time import timeit


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


REQ_VAR = [
    "nature",
    "basin",
    "subbasin",
    "name",
    "storm_speed",
    "storm_dir",
    "usa_pres",
    "usa_rmw",
    "usa_wind",
    "usa_sshs",
    "usa_poci",
    "usa_lat",
    "usa_r34",
    "usa_r50",
    "usa_r64",
    "usa_record",
]


def reduce_to_req(ds: xr.Dataset) -> xr.Dataset:
    """
    Reduce IBTRaCS dataset to the small number
    of variables that are actually used.

    Args:
        ds (xr.Dataset): dataset.

    Returns:
        xr.Dataset: dataset.

    Example::
        >>> from src.data_loading.ibtracs import katrina, reduce_to_req, REQ_VAR
        >>> ds = reduce_to_req(katrina())
        >>> np.all([var in REQ_VAR for var in ds])
        True
    """
    return ds[REQ_VAR]


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


def landings_only(ds: xr.Dataset) -> Optional[xr.Dataset]:
    """
    Extract a reduced dataset based on those points at which a landing occurs.

    Args:
        ds (xr.Dataset): Individual storm.

    Returns:
        Optional[xr.Dataset]: Clipped storm. If there are no tropical cyclones
            hitting the coatline the coastline then None is returned.
    """
    date_times = np.all(
        [(ds["usa_record"].values == b"L"), (ds["usa_sshs"].values > 0)], axis=0
    ).ravel()
    if np.any(date_times):
        return ds.isel(date_time=date_times)
    else:
        return None


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
        if landing_ds is not None:
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

    Taken from ADCIRCpy.

    Args:
        vmax (float): velocity maximum.
        rmax (float): radius of maximum winds.
        neutral_pressure (float): neutral pressure.
        central_pressure (float): central pressure.
        eye_latitude (float): Latitude of eye.

    Returns:
        float: Holland 2010 B parameter.
    """
    air_density = 1.15
    if neutral_pressure <= central_pressure:
        neutral_pressure = np.nan  # central_pressure + 1.0
    fcor = fcor_from_lat(eye_latitude)
    return (vmax**2 + vmax * rmax * fcor * air_density * np.exp(1)) / (
        neutral_pressure - central_pressure
    )


def holland2010(
    radius: float, bs_coeff: float, x_coeff: float, rmax: float, vmax: float
) -> float:
    """
    Holland 2010 function.

    Args:
        radius (float): Radius at a particular point.
        bs_coeff (float): B coefficient Holland2010.
        x_coeff (float): X coefficient Holland2010.
        rmax (float): Radius of maximum wind.
        vmax (float): Velocity of maximum wind.

    Returns:
        float:
    """
    return (
        vmax
        * (((rmax / radius) ** bs_coeff) * np.exp(1 - (rmax / radius) ** bs_coeff))
        ** x_coeff
    )


def holland2010_gen(rmax: float, vmax: float) -> Callable:
    """
    Holland 2010 Generator.

    Args:
        rmax (float): Radius of maximum of wind.
        vmax (float): Value of maximum wind.

    Returns:
        Callable: holland_fit_func(radius: float, bs_coeff: float, x_coeff: float)
    """

    def holland_fit_func(radius: float, bs_coeff: float, x_coeff: float) -> float:
        return holland2010(radius, bs_coeff, x_coeff, rmax, vmax)

    return holland_fit_func


def holland_fitter(
    rmax: float, vmax: float, neutral_pressure: float, rlist: list, vlist: list
) -> Tuple[Callable, np.ndarray]:
    """
    Holland 2010 Fitter.

    Args:
        rmax (float): Radius of maximum wind.
        vmax (float): Velocity maximum.
        neutral_pressure (float): Neutral Pressure.
        rlist (list): Radius list.
        vlist (list): Velocity list.

    Returns:
        Tuple[Callable, np.ndarray]:
    """

    holland2010_loc = holland2010_gen(rmax, vmax)

    def velocity_function_generator(bs_coeff: float, x_coeff: float) -> Callable:
        def velocity_function(radius: float) -> float:
            return holland2010_loc(radius, bs_coeff, x_coeff)

        return velocity_function

    # bs_coeff = holland_B(hurdat)
    # add bounds
    bi = np.finfo(float).eps  # avoid divide by zero
    bf = neutral_pressure
    bounds = (bi, bf)
    bs_coeff = 1.0
    x_coeff = 1.0
    param_guess = [bs_coeff, x_coeff]
    # do curve fitting
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        popt, pcov = optimize.curve_fit(
            holland2010_loc,
            rlist,
            vlist,
            p0=param_guess,
            bounds=bounds,
            method="dogbox",
        )
    # print("[bs_coeff, x_coeff]", popt)
    velocity_function_instance = velocity_function_generator(*popt)
    # radii = np.linspace(bi, bf, num=500)
    # results = np.array([velocity_function_instance(i) for i in radii])
    # print(radii, results)
    return velocity_function_instance, popt


def holland_fitter_usa(ds: xr.Dataset) -> Tuple[Callable, np.ndarray]:
    """
    Holland Fitter USA.

    Args:
        ds (xr.Dataset): xarray dataset.

    Returns:
        Tuple[Callable, np.ndarray]: velocity function, popt
    """
    init_distance_labels = ["usa_r34", "usa_r50", "usa_r64"]
    init_speeds = [int(x.split("r")[1]) for x in init_distance_labels]
    init_distances = [np.mean(ds[var].values) for var in init_distance_labels]
    speeds = []
    distances = []
    for i in range(len(init_speeds)):
        if not np.isnan(init_distances[i]):
            speeds.append(init_speeds[i])
            distances.append(init_distances[i])

    var_names = ["usa_wind", "usa_rmw", "usa_poci"]
    var_list = [ds[var].values for var in var_names]
    # print([ds[var].attrs["units"] for var in var_names + init_distance_labels])
    # print([ds[var].attrs["description"] for var in var_names + distance_labels])
    var_list = [ds[var].values for var in var_names]
    var_list.append(distances)
    var_list.append(speeds)
    # print(var_list)
    velocity_function_instance, popt = holland_fitter(*var_list)
    # print("[bs_coeff, x_coeff]", popt)
    return velocity_function_instance, popt


def holland_b_fit_usa(ds: xr.Dataset) -> np.ndarray:
    """
    Holland B function fit on windspeeds USA.

    Args:
        ds (xr.Dataset): xarray dataset.

    Returns:
        np.ndarray: numpy array.
    """
    b_coeff_list = []
    for i in range(len(ds.date_time.values)):
        _, popt = holland_fitter_usa(ds.isel(date_time=i))
        b_coeff_list.append(popt[0])
    return np.array(b_coeff_list)


def holland_b_usa(ds: xr.Dataset) -> np.ndarray:
    """
    Calculate Holland 2010 Bs parameter using US variables.

    Args:
        ds (xr.Dataset): Individual IBTRACS storm.

    Returns:
        np.ndarray: B parameters.
    """
    var_names = ["usa_wind", "usa_rmw", "usa_poci", "usa_pres", "usa_lat"]
    var_list = [ds[var].values for var in var_names]
    return holland_b(*var_list)


def holland_b_landing_distribution(
    ds: xr.Dataset, sanitize: bool = True, fit=False
) -> np.ndarray:
    """
    Calculate Holland 2010 B parameter distribution.

    Args:
        ds (xr.Dataset): _description_
        sanitize (bool, optional): _description_. Defaults to True.
        fit (bool, optional): Whether to fit to windspeeds at different distances.
            Defaults to False.

    Returns:
        List[float]: Holland B parameters.

    Example::
        >>> holland_b_landing_distribution(katrina()).tolist()
        [175.00548104161092, 140.70336636832965, 143.18867930024024]
        >>> holland_b_landing_distribution(katrina(), fit=True).tolist()
        [0.2363810378430956, 2.220446049250313e-16, 0.20139260140129567]

    """
    output = []
    for storm in ds["storm"].values:
        landing_ds = landings_only(ds.isel(storm=storm))
        if landing_ds is not None:
            if fit:
                output += holland_b_fit_usa(landing_ds).tolist()
            else:
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
    # print(katrina())
    print(holland_b_landing_distribution(katrina(), fit=True).tolist())
    print(holland_b_landing_distribution(katrina(), fit=False).tolist())
