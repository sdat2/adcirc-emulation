"""ERA5 generate."""
import os
import datetime
import numpy as np
import xarray as xr
from sithom.place import Point
from sithom.time import timeit
from src.constants import DATA_PATH, KAT_EX_PATH
from src.data_loading.ecmwf import katrina_netcdf
from src.conversions import pascal_to_millibar


class ERA5Generation:
    def __init__(
        self,
        output_direc: str = os.path.join(DATA_PATH, "kat_h80"),  # string.
        debug: bool = False,
    ) -> None:
        """
        Generate Holland Hurricane Model.

        Args:
            vmax (float, optional): vmax. Defaults to 54.01667.
            point (Point, optional): point to hit. Defaults to NEW_ORLEANS.
            output_direc (str, optional): Output directory. Defaults to os.path.join(DATA_PATH, "kat_h80").
        """
        self.output_direc = output_direc  # string to output direc
        # impact time for katrina.
        self.impact_time = datetime.datetime(year=2005, month=8, day=29, hour=12)
        self.debug = debug  # bool

    def center_from_time(self, time: np.datetime64) -> Point:
        """
        Assumes 111km per degree.

        Args:
            time (numpy.datetime64): time.

        Returns:
            List[float, float]: lon, lat.
        """
        time = datetime.datetime.utcfromtimestamp(
            (time - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, "s")
        )
        time_delta = time - self.impact_time
        distance = time_delta / datetime.timedelta(seconds=1) * self.trans_speed
        return Point(
            self.point.lon + np.sin(np.radians(self.angle)) * distance / 111e3,
            self.point.lat + np.cos(np.radians(self.angle)) * distance / 111e3,
        )

    def run_impact(self) -> None:
        """
        Run Holland model.
        """
        source_direc = KAT_EX_PATH
        invariant_inputs = [
            "fort.14",
            "fort.15",
            "fort.16",
            "fort.22",
            "fort.33",
            "fort.64.nc",
            "fort.73.nc",
            "fort.74.nc",
            "fort.74.nc",
        ]

        adcirc_exe = "/Users/simon/adcirc-swan/adcircpy/exe/adcirc"

        @timeit
        def create_inputs() -> None:
            if not os.path.exists(self.output_direc):
                os.mkdir(self.output_direc)

            for file in invariant_inputs:
                shutil.copy(
                    os.path.join(source_direc, file),
                    os.path.join(self.output_direc, file),
                )

            for forts in [
                ("fort.217", "fort.218"),
                ("fort.221", "fort.222"),
                ("fort.223", "fort.224"),
            ]:
                self.prepare_run(forts)

        @timeit
        def run_adcirc() -> int:
            command = f"cd {self.output_direc} \n {adcirc_exe} > adcirc_log.txt"
            return os.system(command)

        create_inputs()
        assert run_adcirc() == 0
        # output, error = process.communicate()
        # print(output, error)

    @timeit
    def prepare_run(self, forts: Tuple[str]) -> None:
        """
        Prepare run.

        Args:
            forts (Tuple[str]): e.g. ("fort.221", "fort.222")
        """
        da = read_pressures(os.path.join(KAT_EX_PATH, forts[0]))
        vds_list = []
        pds_list = []
        for time in da.time.values:
            vds, pds = self.tc_time_slice(da, time)
            vds_list.append(vds)
            pds_list.append(pds)

        vds = xr.merge(vds_list)
        pda = xr.merge(pds_list)["pressure"]
        if self.debug:
            pda.to_netcdf(os.path.join(self.output_direc, forts[0]) + ".nc")
            print(vds)
            vds.to_netcdf(os.path.join(self.output_direc, forts[1]) + ".nc")
            print(pda)
        print_pressure(pda, os.path.join(self.output_direc, forts[0]))
        print_wsp(vds, os.path.join(self.output_direc, forts[1]))

    def tc_time_slice(
        self, da: xr.DataArray, time: np.datetime64
    ) -> Tuple[xr.Dataset, xr.Dataset]:
        """
        Tropical Cyclone Time Slice.

        Args:
            da (xr.DataArray): dataarray.
            time (np.datetime64): Time to get center from.

        Returns:
            Tuple[xr.Dataset, xr.Dataset]: velocity_ds, pressure_ds.
        """
        center = self.center_from_time(time)
        lons, lats = np.meshgrid(da.lon, da.lat)
        distances = distances_to_points(center, lons, lats)
        angles = angles_to_points(center, lons, lats)
        ds = xr.Dataset(
            data_vars=dict(
                distance=(["time", "lat", "lon"], np.expand_dims(distances, axis=0)),
                angle=(["time", "lat", "lon"], np.expand_dims(angles, axis=0)),
            ),
            coords=dict(
                lat=(["lat"], da.lat.values),
                lon=(["lon"], da.lon.values),
                time=(["time"], [time]),
            ),
        )
        ds.distance.attrs = {"units": "meters", "long_name": "Distance from center"}
        ds.angle.attrs = {"units": "degrees", "long_name": "Angle from center"}

        windspeed = self.symetric_model.velocity(
            ds.distance.values
        )  # self.windspeed_at_points(lats, lons, point)
        angles = np.radians(ds.angle.values - 90.0)
        u10, v10 = -np.sin(angles) * windspeed, -np.cos(angles) * windspeed
        pressure = pascal_to_millibar(self.symetric_model.pressure(ds.distance.values))
        pds = xr.Dataset(
            data_vars=dict(
                pressure=(["time", "lat", "lon"], pressure),
            ),
            coords=dict(
                lat=(["lat"], da.lat.values),
                lon=(["lon"], da.lon.values),
                time=(["time"], [time]),
            ),
        )
        pds.pressure.attrs = {"units": "mb", "long_name": "Surface pressure"}
        vds = xr.Dataset(
            data_vars=dict(
                U10=(["time", "lat", "lon"], u10),
                V10=(["time", "lat", "lon"], v10),
            ),
            coords=dict(
                lat=(["lat"], da.lat.values),
                lon=(["lon"], da.lon.values),
                time=(["time"], [time]),
            ),
        )
        vds.U10.attrs = {"units": "m s**-1", "long_name": "Zonal velocity"}
        vds.V10.attrs = {"units": "m s**-1", "long_name": "Meridional velocity"}
        return vds, pds
