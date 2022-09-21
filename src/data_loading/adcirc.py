"""ADCIRC."""
import os
import xarray as xr
from src.constants import KAT_EX_PATH
import netCDF4 as nc


if __name__ == "__main__":
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
