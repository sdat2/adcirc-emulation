"""Download tidal guages."""
from typing import List  # import pandas as pd
import xarray as xr
from dateutil import parser
from src.constants import NEW_ORLEANS, DEFAULT_GAUGES, KATRINA_TIDE_NC

try:
    from noaa_coops.noaa_coops import stationid_from_bbox, Station
except ImportError:
    print("no noaa_coops available")


def filter_by_age(stationid_list: List[str], date: str = "2005-06-01") -> List[str]:
    """
    Filter the stations to be active before the date given.

    Args:
        stationid_list (List[str]): list of stations.
        date (optional, str): Date to go after.
    """
    station_list = []
    station_id_list = []

    for stationid in stationid_list:
        station = Station(stationid)
        print(
            stationid,
            f"After {date}::\t",
            is_after(date, station.metadata["details"]["origyear"]),
        )

        if is_after(date, station.metadata["details"]["origyear"]):
            station_list.append(station)
            station_id_list.append(stationid)

    return station_id_list


def katrina_data(stationid_list: List[str] = DEFAULT_GAUGES) -> xr.Dataset:
    """Return Katrina Data.

    Args:
        stationid_list (List[str], optional): Stationid list. Defaults to DEFAULT_GAUGES.

    Returns:
        xr.Dataset: Gauge data for Katrina including latitude, longitude, and name.
    """
    data_list = []
    for stationid in stationid_list:
        station = Station(stationid)
        try:
            data = station.get_data(
                begin_date="20050820",
                end_date="20050902",
                product="water_level",
                datum="MSL",  # "MLLW",
                units="metric",
                time_zone="gmt",
            )
            xr_data = (
                data.to_xarray()
                .expand_dims(dim="stationid")
                .assign_coords(coords={"stationid": (["stationid"], [stationid])})
            )
            xr_data = xr_data.assign_coords(
                coords={
                    "lon": (["stationid"], [station.lat_lon["lon"]]),
                    "lat": (["stationid"], [station.lat_lon["lat"]]),
                    "name": (["stationid"], [station.metadata["name"]]),
                }
            )

            data_list.append(xr_data)  # .expand_dims(dim="stationid")
        except Exception as e:
            print(stationid, "problem", e)
    return xr.merge(data_list)


def save_katrina_nc(stationid_list: List[str] = DEFAULT_GAUGES) -> None:
    """Katrina tidal data.

    Args:
        stationid_list (List[str], optional): Stationid list.
            Defaults to DEFAULT_GAUGES.
    """
    katrina_data(stationid_list).to_netcdf(KATRINA_TIDE_NC)


def is_after(time_a: str, time_b: str) -> bool:
    """
    Is time_a after time_b?

    Args:
        time_a (str): First time string
        time_b (str): Second time string.

    Returns:
        bool: the answer.
    """
    time_a = parser.parse(time_a)
    time_b = parser.parse(time_b)
    # print("time_a", time_a)
    # print("time_b", time_b)
    return time_a > time_b


if __name__ == "__main__":
    # print(stationid_from_bbox([-74.4751,40.389,-73.7432,40.9397]))
    # for product in station.metadata["products"]["products"]:
    #     print(product["name"])
    # python src/data_loading/tides.py
    # save_katrina_nc()
    save_katrina_nc(filter_by_age(stationid_from_bbox(NEW_ORLEANS.bbox())))
