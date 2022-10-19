"""ERA5 generate."""
from typing import Tuple
import os
import shutil
import datetime
import numpy as np
import xarray as xr
from sithom.place import Point
from sithom.time import timeit
from src.constants import DATA_PATH, KAT_EX_PATH
from src.data_loading.ecmwf import katrina_netcdf
from src.data_loading.adcirc import (
    print_pressure,
    print_wsp,
    read_pressures,
    read_windspeeds,
)
from src.conversions import pascal_to_millibar, distances_to_points, angles_to_points


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
