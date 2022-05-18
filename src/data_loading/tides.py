"""Tides"""
from noaa_coops.noaa_coops import stationid_from_bbox


def bbox_from_loc(loc=[-90.0715, 29.9511], buffer: float=1):
    return [loc[0] - buffer, loc[1] - buffer, loc[0] + buffer, loc[1] + buffer]

if __name__ == "__main__":
    print(stationid_from_bbox([-74.4751,40.389,-73.7432,40.9397]))
    print(stationid_from_bbox(bbox_from_loc()))

