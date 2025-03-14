"""ADCIRC Input reading (and writing).


https://coast.nd.edu/reports_papers/SELA_2007_IDS_2_FinalDraft/App%20D%20PBL-C%20WIN_PRE%20File%20Format.pdf

Currently using OWI text format.

"""
from typing import List, Union, Optional
import os
import datetime
import numpy as np
import xarray as xr
from src.constants import DATA_PATH, KAT_EX_PATH
import netCDF4 as nc
from adcircpy.outputs import Maxele, Fort63
from sithom.place import BoundingBox
from src.constants import KAT_EX_PATH, NEW_ORLEANS
from src.preprocessing.sel import trim_tri


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
def datetime_to_int(date: Union[datetime.datetime, np.datetime64]) -> int:
    """
    Datetime.

    Args:
        date (Union[datetime.datetime, np.datetime64]):

    Returns:
        int: int to output.
    """
    if isinstance(date, np.datetime64):
        date = datetime.datetime.utcfromtimestamp(
            (date - np.datetime64("1970-01-01T00:00:00Z")) / np.timedelta64(1, "s")
        )
    return int(
        str(date.year)
        + two_char_int(date.month)
        + two_char_int(date.day)
        + two_char_int(date.hour)
        + two_char_int(date.minute)
    )


def read_owi_data_line(line: str) -> List[float]:
    """
    Read data line.

    Args:
        line (str): Read a line of data.

    Returns:
        List[float]: _description_
    """
    # 4 decimal point figures
    return [float(x) for x in line.strip("\n").split(" ") if x != ""]


def read_owi_coord_line(line, names) -> dict:
    """
    Process line.

    Example::
        >>> names = ["iLat", "iLong", "DX", "DY", "SWLat", "SWLon", "DT"]
        >>> line = "iLat=  46iLong=  60DX=0.0500DY=0.0500SWLat=28.60000SWLon=-90.2800DT=200508250000"
        >>> read_owi_coord_line(line, names)
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


def read_owi_windspeeds(windspeed_path: str) -> xr.Dataset:
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
        # print("len pressure", len_wsp)

        # print(wsp_list[0])
        # print (wsp_list[1])
        # print (wsp_list[2])
        # print (read_owi_data_line(wsp_list[2]))

        wsp_lol.append([])
        for i in range(2, len_wsp):
            if wsp_list[i].startswith(" "):
                wsp_lol[-1].append(read_owi_data_line(wsp_list[i]))
            else:
                # print (wsp_list[i])
                wsp_lol.append([])

        names = ["iLat", "iLong", "DX", "DY", "SWLat", "SWLon", "DT"]
        coords = read_owi_coord_line(wsp_list[1], names)
        dates = int_to_datetime(
            np.array(
                [
                    read_owi_coord_line(x, names)["DT"]
                    for x in wsp_list
                    if x.startswith("i")
                ]
            ).astype(int)
        )
        lats = np.array(
            [coords["SWLat"] + coords["DY"] * i for i in range(int(coords["iLat"]))]
        )
        lons = np.array(
            [coords["SWLon"] + coords["DX"] * i for i in range(int(coords["iLong"]))]
        )
        data = np.array(wsp_lol).reshape(len(dates), 2, len(lats), len(lons))
        ds = xr.Dataset(
            data_vars=dict(
                U10=(["time", "lat", "lon"], data[:, 0, :, :], {"units": "m s**-1", "long_name": "Zonal 10m windspeed"}),
                V10=(["time", "lat", "lon"], data[:, 1, :, :], {"units": "m s**-1", "long_name": "Meridional 10m windspeed"}),
            ),
            coords=dict(
                lon=(["lon"], lons, {"units": "degree_East", "long_name": "Longitude"}),
                lat=(["lat"], lats, {"units": "degree_North", "long_name": "Latitude"}),
                time=dates,
            ),
            attrs=dict(
                description="Velocities could be the wrong way round",
                grid_var=str(coords),
            ),
        )
        return ds


def read_owi_pressures(pressure_path: str) -> xr.DataArray:
    """
    Read pressures.

    Args:
        pressure_path (str): Path to pressure fort file.

    Returns:
        xr.DataArray: _description_
    """
    with open(pressure_path) as file:

        pressure_list = [x for x in file]
        pressure_lol = []

        len_pressure = len(pressure_list)

        names = ["iLat", "iLong", "DX", "DY", "SWLat", "SWLon", "DT"]
        coords = read_owi_coord_line(pressure_list[1], names)

        pressure_lol.append([])
        for i in range(2, len_pressure):
            if pressure_list[i].startswith(" "):
                pressure_lol[-1].append(read_owi_data_line(pressure_list[i]))
            else:
                # print (pressure_list[i])
                pressure_lol.append([])

        dates = int_to_datetime(
            np.array(
                [
                    read_owi_coord_line(x, names)["DT"]
                    for x in pressure_list
                    if x.startswith("i")
                ]
            ).astype(int)
        )
        lats = np.array(
            [coords["SWLat"] + coords["DY"] * i for i in range(int(coords["iLat"]))]
        )
        lons = np.array(
            [coords["SWLon"] + coords["DX"] * i for i in range(int(coords["iLong"]))]
        )

        # 56 diff list - one for each timestep.
        da = xr.DataArray(
            data=np.array(pressure_lol).reshape(len(dates), len(lats), len(lons)),
            dims=["time", "lat", "lon"],
            coords=dict(
                lon=(["lon"], lons, {"units": "degree_East", "long_name": "Longitude"}),
                lat=(["lat"], lats, {"units": "degree_North", "long_name": "Latitude"}),
                time=dates,
            ),
            attrs=dict(
                long_name="Pressure",
                description="Surface pressure",
                units="mb",
            ),
        )
        return da


def read_owi_default_inputs() -> None:
    for file_tuple in [
        ("fort.217", "fort.218"),
        ("fort.221", "fort.222"),
        ("fort.223", "fort.224"),
    ]:
        pr_ds = read_owi_pressures(os.path.join(KAT_EX_PATH, file_tuple[0]))
        pr_ds.to_netcdf(os.path.join(DATA_PATH, file_tuple[0]) + ".nc")
        write_owi_pressures(
            pr_ds,
            os.path.join(DATA_PATH, file_tuple[0]),
        )
        ws_ds = read_owi_windspeeds(os.path.join(KAT_EX_PATH, file_tuple[1]))
        ws_ds.to_netcdf(os.path.join(DATA_PATH, file_tuple[1]) + ".nc")
        write_owi_windspeeds(
            ws_ds,
            os.path.join(DATA_PATH, file_tuple[1]),
        )


# windspeed 8 entries, 3 s.f. 3 space
def entry(inp: float) -> str:
    """
    Input.

    Args:
        inp (float): input float.

    Returns:
        str: 10 character string for fortran input (4 decimal places).
    """
    tot_len = 6 + 4
    num = "{:.4f}".format(inp)
    spaces = tot_len - len(num)
    return " " * spaces + num


def make_line(inp: List[float]) -> str:
    """
    Make line.

    Args:
        inp (List[float]): input.

    Returns:
        str: line of floats.
    """
    return "".join(list(map(entry, inp)))


def write_owi_pressures(da: xr.DataArray, output_path: str) -> None:
    """
    Print pressure text files.

    Args:
        da (xr.DataArray): da.
        output_path (str): output path.
    """
    ds = str(datetime_to_int(da.time.values[0]))[:-2]
    de = str(datetime_to_int(da.time.values[-1]))[:-2]
    lats = da.lat.values
    lons = da.lon.values
    swlat = "{:.5f}".format(np.min(lats))
    swlon = "{:.4f}".format(np.min(lons))
    dy = "{:.4f}".format(lats[1] - lats[0])
    dx = "{:.4f}".format(lons[1] - lons[0])
    default_len = 4
    ilon = str(len(lons))
    ilon = " " * (default_len - len(ilon)) + ilon
    ilat = str(len(lats))
    ilat = " " * (default_len - len(ilat)) + ilat
    first_line = f"Oceanweather WIN/PRE Format                            {ds}     {de}"

    with open(output_path, "w") as file:
        # print (first_line)
        file.write(first_line + "\n")

        for time in da.time.values:
            dt = str(datetime_to_int(time))
            data = list(
                da.sel(time=time).values.reshape(int(len(lons) * len(lats) / 8), 8)
            )
            data_list_str = [make_line(float_line) for float_line in data]
            date_line = str(
                f"iLat={ilat}iLong={ilon}DX={dx}DY={dy}"
                + f"SWLat={swlat}SWLon={swlon}DT={dt}"
            )
            file.write(date_line + "\n")
            for line in data_list_str:
                file.write(line + "\n")

    # iLat=  46iLong=  60DX=0.0500DY=0.0500SWLat=28.60000SWLon=-90.2800DT=200508250000


def write_owi_windspeeds(wds: xr.Dataset, output_path: str) -> None:
    """
    Print windspeed.

    Args:
        wds (xr.Dataset): windspeed dataset.
        output_path (str): output path.
    """
    # wds = xr.open_dataset(input_path)
    # header does not include minutes.
    ds = str(datetime_to_int(wds.time.values[0]))[:-2]
    de = str(datetime_to_int(wds.time.values[-1]))[:-2]
    lats = wds.lat.values
    lons = wds.lon.values
    swlat = "{:.5f}".format(np.min(lats))
    swlon = "{:.4f}".format(np.min(lons))
    dy = "{:.4f}".format(lats[1] - lats[0])
    dx = "{:.4f}".format(lons[1] - lons[0])
    default_len = 4
    ilon = str(len(lons))
    ilon = " " * (default_len - len(ilon)) + ilon
    ilat = str(len(lats))
    ilat = " " * (default_len - len(ilat)) + ilat
    first_line = f"Oceanweather WIN/PRE Format                            {ds}     {de}"

    with open(output_path, "w") as file:
        file.write(first_line + "\n")
        for time in wds.time.values:

            dt = str(datetime_to_int(time))
            data_u10 = list(
                wds.U10.sel(time=time).values.reshape(int(len(lons) * len(lats) / 8), 8)
            )
            data_v10 = list(
                wds.V10.sel(time=time).values.reshape(int(len(lons) * len(lats) / 8), 8)
            )
            data = data_u10 + data_v10
            data_list_str = [make_line(float_line) for float_line in data]
            date_line = f"iLat={ilat}iLong={ilon}DX={dx}DY={dy}SWLat={swlat}SWLon={swlon}DT={dt}"
            file.write(date_line + "\n")
            for line in data_list_str:
                file.write(line + "\n")


def main():
    # python src/data_loading/adcirc.py
    nc_files = [x for x in os.listdir(KAT_EX_PATH) if x.endswith(".nc")]
    # print (KAT_EX_PATH)
    for file in nc_files:
        # print (file)

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


def select_coastal_cells(
    lon: float, lat: float, number: int = 10
) -> Optional[xr.Dataset]:
    """
    Select coastal cells.

    Args:
        lon (float): Longitude of central point.
        lat (float): Latitude of central point.
        number (int, optional): How many to choose initially. Defaults to 10.

    Returns:
        Optional[xr.Dataset]: coastal sealevel point dataset.
    """
    # me = Maxele(os.path.join(KAT_EX_PATH, "maxele.63.nc"))
    f63 = Fort63(os.path.join(KAT_EX_PATH, "fort.63.nc"))
    lats = f63.y.copy()
    lons = f63.x.copy()
    index_list = []
    for _ in range(number):
        index = ((lons - lon) ** 2 + (lats - lat) ** 2).argmin()
        # print("index", index)
        # print(lons[index], lats[index])
        index_list.append(index)
        lons[index] = -100
        lats[index] = -100

    (uniq, freq) = np.unique(f63.triangles, return_counts=True)
    coastals = uniq[freq <= 4]
    indices = np.array(index_list)
    print("Coastals", coastals, len(coastals))
    print("Indices", indices, len(indices))

    indices = indices[[x in coastals.tolist() for x in index_list]]

    print(indices)
    if len(indices) > 0:
        print(f63.x.shape, f63.y.shape, f63._ptr["zeta"].shape)
        lats = f63.y[indices]
        lons = f63.y[indices]
        heights = f63._ptr["zeta"][:, indices]
        start = datetime.datetime(year=2005, month=8, day=19, hour=5)
        time_step = datetime.timedelta(hours=1, minutes=20)
        times = [start + i * time_step for i in range(heights.shape[0])]
        ds = xr.Dataset(  #
            data_vars=dict(Height=(["time", "point"], heights)),
            coords=dict(
                lon=(["point"], lons),
                lat=(["point"], lats),
                time=times,
            ),
        )
        ds["Height"].attrs["units"] = "m"
        return ds
    else:
        return ds


def timeseries_height_ds(
    path: str = KAT_EX_PATH, bbox: BoundingBox = NEW_ORLEANS.bbox(3)
) -> xr.Dataset:
    """
    Open the fort.63.nc file in the path, read the contents, and get the dataset out.
    """
    f63 = Fort63(os.path.join(path, "fort.63.nc"))
    # Trim triangles
    x, y, tri, z = trim_tri(f63.x, f63.y, f63.triangles, bbox, f63._ptr["zeta"][:])
    start = datetime.datetime(year=2005, month=8, day=19, hour=5)
    time_step = datetime.timedelta(hours=1, minutes=20)
    times = [start + i * time_step for i in range(z.shape[0])]
    ds = xr.Dataset(
        data_vars={
            "zeta": (["time", "point"], z.data),
            "mesh": (["triangle", "vertex"], tri),
        },
        coords={
            "lon": (["point"], x),
            "lat": (["point"], y),
            "time": (["time"], times),
        },
    )
    ds["time"].attrs["long_name"] = "Time"
    ds["zeta"].attrs["units"] = "m"
    ds["zeta"].attrs["long_name"] = "Sea Surface Height"
    ds["mesh"].attrs["units"] = "dimensionless"
    ds["mesh"].attrs["long_name"] = "ADCIRC Mesh"
    ds["lon"].attrs["units"] = "degrees_East"
    ds["lat"].attrs["units"] = "degrees_North"
    ds.attrs["BoundingBox"] = str(bbox)
    ds.zeta.values[ds.zeta.values == -99999.0] = 0
    return ds


if __name__ == "__main__":
    # python src/data_loading/adcirc.py
    read_owi_default_inputs()
    # test()
