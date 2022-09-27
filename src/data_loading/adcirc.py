"""ADCIRC Input reading."""
from typing import List
import os
import datetime
import numpy as np
import xarray as xr
from src.constants import DATA_PATH, KAT_EX_PATH
import netCDF4 as nc


@np.vectorize
def int_to_datetime(int_input: int) -> datetime.datetime:
    """
    Int format to datetime.

    Args:
        int_input (int): Int input.

    Returns:
        datetime.datetime: Datetime.
    """
    date_str = str(int_input)
    return datetime.datetime(
        year=int(date_str[:4]),
        month=int(date_str[4:6]),
        day=int(date_str[6:8]),
        hour=int(date_str[8:10]),
        minute=int(date_str[10:12]),
    )


def two_char_int(int_input: int) -> str:
    """
    Two char int.

    Args:
        int_input (int): input integer.

    Returns:
        str: Two char int.
    """
    ret_str = str(int_input)
    if len(ret_str) == 1:
        ret_str = "0" + ret_str
    return ret_str


@np.vectorize
def datetime_to_int(date: datetime.datetime) -> int:
    """
    Datetime.

    Args:
        date (datetime.datetime):

    Returns:
        int: int to output.
    """
    return int(
        str(date.year)
        + two_char_int(date.month)
        + two_char_int(date.day)
        + two_char_int(date.hour)
        + two_char_int(date.minute)
    )


def read_data_line(line: str) -> List[float]:
    """
    Read data line.

    Args:
        line (str): _description_

    Returns:
        List[float]: _description_
    """
    # 4 decimal point figures
    return [float(x) for x in line.strip("\n").split(" ") if x != ""]


def read_coord_line(line, names) -> dict:
    """
    Process line.

    Example::
        >>> names = ["iLat", "iLong", "DX", "DY", "SWLat", "SWLon", "DT"]
        >>> line = "iLat=  46iLong=  60DX=0.0500DY=0.0500SWLat=28.60000SWLon=-90.2800DT=200508250000"
        >>> read_coord_line(line, names)
        {'iLat': 46.0,
         'iLong': 60.0,
         'DX': 0.05,
         'DY': 0.05,
         'SWLat': 28.6,
         'SWLon': -90.28,
         'DT': 200508250000.0}
    """
    result_dict = {}

    for i, name in enumerate(names):
        line = line.strip(name + "=")
        if i < len(names) - 1:
            int_l = line.split(names[i + 1] + "=")
            result_dict[name] = float(int_l[0])
            line = int_l[1]
        else:
            result_dict[name] = float(int_l[1])

    return result_dict


def read_windspeeds(windspeed_path: str) -> xr.Dataset:
    """
    Read windspeeds.

    Args:
        windspeed_path (str): Windspeed paths.

    Returns:
        xr.Dataset: uvel, vvel variables.
    """
    with open(windspeed_path) as file:

        wsp_list = [x for x in file]
        wsp_lol = []

        len_wsp = len(wsp_list)
        print("len pressure", len_wsp)

        print(wsp_list[0])
        print(wsp_list[1])
        print(wsp_list[2])
        print(read_data_line(wsp_list[2]))

        wsp_lol.append([])
        for i in range(2, len_wsp):
            if wsp_list[i].startswith(" "):
                wsp_lol[-1].append(read_data_line(wsp_list[i]))
            else:
                print(wsp_list[i])
                wsp_lol.append([])

        names = ["iLat", "iLong", "DX", "DY", "SWLat", "SWLon", "DT"]
        coords = read_coord_line(wsp_list[1], names)
        dates  = int_to_datetime(np.array([
            read_coord_line(x, names)["DT"] for x in wsp_list if x.startswith("i")
        ]).astype(int))
        lats = np.array(
            [coords["SWLat"] + coords["DY"] * i for i in range(int(coords["iLat"]))]
        )
        lons = np.array(
            [coords["SWLon"] + coords["DX"] * i for i in range(int(coords["iLong"]))]
        )
        data = np.array(wsp_lol).reshape(len(dates), 2, len(lats), len(lons))
        return xr.Dataset(
            data_vars=dict(
                uvel=(["time", "lat", "lon"], data[:, 0, :, :]),
                vvel=(["time", "lat", "lon"], data[:, 1, :, :]),
            ),
            coords=dict(
                lon=(["lon"], lons),
                lat=(["lat"], lats),
                time=dates,
            ),
            attrs=dict(description="Velocities could be the wrong way round"),
        )


def read_pressures(pressure_path: str) -> xr.DataArray:
    """
    Read pressures.

    Args:
        pressure_path (str): _description_

    Returns:
        xr.DataArray: _description_
    """
    with open(pressure_path) as file:

        pressure_list = [x for x in file]
        pressure_lol = []

        len_pressure = len(pressure_list)
        print("len pressure", len_pressure)

        print(pressure_list[0])
        print(pressure_list[1])
        print(pressure_list[2])
        print(read_data_line(pressure_list[2]))

        names = ["iLat", "iLong", "DX", "DY", "SWLat", "SWLon", "DT"]
        coords = read_coord_line(pressure_list[1], names)

        pressure_lol.append([])
        for i in range(2, len_pressure):
            if pressure_list[i].startswith(" "):
                pressure_lol[-1].append(read_data_line(pressure_list[i]))
            else:
                print(pressure_list[i])
                pressure_lol.append([])

        dates  = int_to_datetime(np.array([
            read_coord_line(x, names)["DT"] for x in pressure_list if x.startswith("i")
        ]).astype(int))
        lats = np.array(
            [coords["SWLat"] + coords["DY"] * i for i in range(int(coords["iLat"]))]
        )
        lons = np.array(
            [coords["SWLon"] + coords["DX"] * i for i in range(int(coords["iLong"]))]
        )

        # 56 diff list - one for each timestep.
        return xr.DataArray(
            data=np.array(pressure_lol).reshape(len(dates), len(lats), len(lons)),
            dims=["time", "lat", "lon"],
            coords=dict(
                lon=(["lon"], lons),
                lat=(["lat"], lats),
                time=dates,
            ),
            attrs=dict(
                description="Pressure",
                long_name="Pressure",
                description="Surface pressure",
                units="mb",
            ),
        )


def read_default_inputs() -> None:
    for file_tuple in [
        ("fort.217", "fort.218"),
        ("fort.221", "fort.222"),
        ("fort.223", "fort.224"),
    ]:
        pr_ds = read_pressures(os.path.join(KAT_EX_PATH, file_tuple[0]))
        pr_ds.to_netcdf(os.path.join(DATA_PATH, file_tuple[0]) + ".nc")
        ws_ds = read_windspeeds(os.path.join(KAT_EX_PATH, file_tuple[1]))
        ws_ds.to_netcdf(os.path.join(DATA_PATH, file_tuple[1]) + ".nc")


def main():
    # python src/data_loading/adcirc.py
    nc_files = [x for x in os.listdir(KAT_EX_PATH) if x.endswith(".nc")]
    print(KAT_EX_PATH)
    for file in nc_files:
        print(file)

        try:
            print(
                xr.open_dataset(
                    os.path.join(KAT_EX_PATH, file),
                    engine="netcdf4",
                    decode_cf=False,
                    decode_coords=False,
                    decode_timedelta=False,
                )
            )
        except Exception as e:
            print(e)

        try:
            nc_ds = nc.Dataset(os.path.join(KAT_EX_PATH, file))
            print(nc_ds.variables)
            for var in nc_ds.variables.values():
                print(var)

        except Exception as e:
            print(e)


if __name__ == "__main__":
    # python src/data_loading/adcirc.py
    read_default_inputs()
