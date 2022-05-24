"""Download tidal guages."""
from typing import List
from dateutil import parser
from noaa_coops.noaa_coops import stationid_from_bbox, Station
from src.constants import NEW_ORLEANS


GUAGES = ['8761724', '8761927', '8761955', '8762075', '8762482']

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
    station_list = []
    station_id_list = []

    for stationid in stationid_list:
        station = Station(stationid)
        # print(station)
        print("After 2005::\t",
              is_after("2005", station.metadata["details"]["origyear"]))

        if is_after("2005", station.metadata["details"]["origyear"]):
           station_list.append(station)
           station_id_list.append(stationid)

    print(station_list)
    print(station_id_list)

    return station_list


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
    #print(stationid_from_bbox([-74.4751,40.389,-73.7432,40.9397]))
    print_station_details(stationid_from_bbox(bbox_from_loc()))
    #print(bbox_from_loc())
    station = Station(8762483)
    # print(station.metadata)
    # for key in station.metadata:
    #     print(key, "::", station.metadata[key])

    print(station.metadata["id"],
          station.metadata["name"],
          station.metadata["lng"],
          station.metadata["lat"],
          station.metadata["details"]["origyear"],
          type(station.metadata["details"]["origyear"])
          )

    print(is_after(station.metadata["details"]["origyear"], "2016"))
    print(is_after(station.metadata["details"]["origyear"], "2014"))
    print(is_after("2005", station.metadata["details"]["origyear"]))

    # for product in station.metadata["products"]["products"]:
    #     print(product["name"])
    # python src/data_loading/tides.py
