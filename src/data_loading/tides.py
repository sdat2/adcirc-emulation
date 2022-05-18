"""Download tidal guages."""
from typing import List
from noaa_coops.noaa_coops import stationid_from_bbox, Station
from src.constants import NEW_ORLEANS

def bbox_from_loc(loc: List[float]=NEW_ORLEANS, buffer: float=1) -> List[float]:
    """
    Get bbox padding a central location.

    Args:
        loc (List[float], optional): [Lon, Lat]. Defaults to NEW_ORLEANS.
        buffer (float, optional): How many degrees to go out from loc. Defaults to 1.

    Returns:
        List[Float]: A bounding box like [-91.0715, 28.9511, -89.0715, 30.9511].
    """
    return [loc[0] - buffer, loc[1] - buffer, loc[0] + buffer, loc[1] + buffer]

def print_station_details(stationid_list: List[str]) -> None:
    """
    Print the station details.

    Args:
        stationid_list (List[str]): list of stations
    """
    for stationid in stationid_list:
        print(Station(stationid), "\n \n")


if __name__ == "__main__":
    #print(stationid_from_bbox([-74.4751,40.389,-73.7432,40.9397]))
    #print_station_details(stationid_from_bbox(bbox_from_loc()))
    #print(bbox_from_loc())
    station = Station(8762483)
    print(station.metadata["details"]["origyear"])
    # print(station.metadata)
    for key in station.metadata:
        print(key, "::", station.metadata[key])

    print(station.metadata["id"],
    station.metadata["name"], station.metadata["lng"],
    station.metadata["lat"], station.metadata["details"]["origyear"])
    for product in station.metadata["products"]["products"]:
        print(product["name"])



