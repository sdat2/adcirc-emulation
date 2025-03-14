"""Generate hurricane."""
import os
import shutil
from typing import Tuple, List
import datetime
import difflib
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from src.constants import (
    DATA_PATH,
    FIGURE_PATH,
    KAT_EX_PATH,
    NEW_ORLEANS,
    NO_BBOX,
    ADCIRC_EXE,
)
from sithom.plot import plot_defaults, label_subplots
from sithom.place import Point
from sithom.time import timeit
from src.conversions import (
    distances_to_points,
    angles_to_points,
    millibar_to_pascal,
    nmile_to_meter,
    pascal_to_millibar,
)
from src.data_loading.adcirc import (
    write_owi_pressures,
    write_owi_windspeeds,
    read_owi_pressures,
    read_owi_windspeeds,
)
from src.models.h08 import h08_vp


class TropicalCyclone:
    def __init__(
        self,
        point: Point,
        angle: float,
        trans_speed: float,
    ) -> None:
        """
        Tropical cylone to hit coast at point.

        Args:
            point (Point): Point to impact (lon, lat).
            angle (float): Angle to point [degrees].
            trans_speed (float): Translation speed [m s**-1].
        """
        # print(angle, trans_speed)
        self.point = point
        self.angle = angle
        self.trans_speed = trans_speed
        self.time_delta = datetime.timedelta(hours=3)
        self.impact_time = datetime.datetime(year=2005, month=8, day=29, hour=12)

    def __repr__(self) -> str:
        return str(
            "point: "
            + str(self.point)
            + "\n"
            + "angle: "
            + str(self.angle)
            + " degrees\n"
            + "trans_speed: "
            + str(self.trans_speed)
            + " ms-1\n"
            + "vmax: "
            + str(self.vmax)
            + " ms-1\n"
            + "rmax: "
            + str(self.rmax)
            + " km\n"
            + "bs: "
            + str(self.bs)
            + " units\n"
        )

    def new_point(self, distance: float) -> List[float]:
        """
        Line. Assumes 111km per degree.

        Args:
            distance (float): Distance in meters.

        Returns:
            List[float, float]: lon, lat.
        """
        return [
            self.point.lon + np.sin(np.radians(self.angle)) * distance / 111e3,
            self.point.lat + np.cos(np.radians(self.angle)) * distance / 111e3,
        ]

    def trajectory(self, run_up=1e6, run_down=3.5e5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Trajectory.

        Args:
            run_up (int, optional): Run up afterwards. Defaults to 1000 km in meteres.
            run_down (int, optional): Run down after point. Defaults to 350 km im meters.
        """
        distance_per_timestep = (
            self.trans_speed * self.time_delta / datetime.timedelta(seconds=1)
        )
        time_steps_before = int(abs(run_up) / distance_per_timestep)
        time_steps_after = int(abs(run_down) / distance_per_timestep)
        # print(self.point, self.angle, run_up, run_down)
        point_list = [
            self.new_point(dist)
            for dist in range(-int(run_up), int(run_down), int(distance_per_timestep))
        ]
        time_list = [
            self.impact_time + x * self.time_delta
            for x in range(
                -time_steps_before,
                time_steps_after + 1,
                1,
            )
        ]
        print(time_steps_before + time_steps_after + 1)
        return np.array(point_list), np.array(time_list)

    # def time_traj(self, )
    def trajectory_ds(self, run_up=1e6, run_down=3.5e5) -> xr.Dataset:
        """
        Create a trajectory dataset for the center eye of the tropical cylone.

        Args:
            run_up (float, optional): How many meters to run up. Defaults to 1e6.
            run_down (float, optional): How many meters to run down. Defaults to 3.5e5.

        Returns:
            xr.Dataset: trajectory dataset with variables lon, lat and time.
        """
        traj, dates = self.trajectory(run_up=run_up, run_down=run_down)
        print(traj.shape)
        print(dates.shape)
        return xr.Dataset(
            data_vars=dict(
                lon=(["time"], traj[:, 0]),
                lat=(["time"], traj[:, 1]),
            ),
            coords=dict(
                time=dates,
                # reference_time=self.impact_time,
            ),
            attrs=dict(description="Tropcial Cylone trajectory."),
        )

    def angle_at_points(
        self, lats: np.ndarray, lons: np.ndarray, point: Point
    ) -> np.ndarray:
        """
        Angles from each point.

        Args:
            lats (np.ndarray): Latitudes.
            lons (np.ndarray): Longitudes.
            point (Point): Point around which to go.

        Returns:
            np.ndarray: Angles in degrees from North.
        """
        return angles_to_points(point, lons, lats)

    def velocity_at_points(
        self, lats: np.ndarray, lons: np.ndarray, point: Point
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Velocity at points.

        Args:
            lats (np.ndarray): Latitudes [degrees_North].
            lons (np.ndarray): Longitudes [degrees_East].
            point (Point): point (lon, lat).

        Returns:
            Tuple[np.ndarray, np.ndarray]: u_vel [m s**-1], v_vel [m s**-1]
        """
        windspeed = self.windspeed_at_points(lats, lons, point)
        angle = np.radians(self.angle_at_points(lats, lons, point) - 90.0)
        return np.sin(angle) * windspeed, np.cos(angle) * windspeed


def mult_folder_name(mult: int) -> str:
    """
    Args:
        mult (int): Multiply.

    Returns:
        str: folder path.
    """

    return os.path.join(DATA_PATH, "mult" + str(mult))


def mult_generation(mult: int = 1) -> None:
    """
    Multiply Katrina by 2 for new example.
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
    pressure_inputs = [
        "fort.217",
        "fort.221",
        "fort.223",
    ]
    wsp_inputs = [
        "fort.218",
        "fort.222",
        "fort.224",
    ]

    output_direc = mult_folder_name(mult)
    adcirc_exe = "/Users/simon/adcirc-swan/adcircpy/exe/adcirc"

    @timeit
    def create_inputs() -> None:
        if not os.path.exists(output_direc):
            os.mkdir(output_direc)

        for file in invariant_inputs:
            shutil.copy(
                os.path.join(source_direc, file), os.path.join(output_direc, file)
            )

        for file in pressure_inputs:
            orginal_file = os.path.join(source_direc, file)
            ds = read_owi_pressures(orginal_file)
            final_file = os.path.join(output_direc, file)
            write_owi_pressures(ds, final_file)

        for file in wsp_inputs:
            orginal_file = os.path.join(source_direc, file)
            ds = read_owi_windspeeds(orginal_file)
            final_file = os.path.join(output_direc, file)
            ds = ds * mult
            write_owi_windspeeds(ds, final_file)

    @timeit
    def run_adcirc() -> int:
        command = f"cd {output_direc} \n {adcirc_exe} > adcirc_log.txt"
        return os.system(command)

    create_inputs()
    assert run_adcirc() == 0
    # output, error = process.communicate()
    # print(output, error)


def comp() -> None:
    """
    Compare the wind and pressure files.
    """

    pres_files = ["fort.217", "fort.221", "fort.223"]
    wind_files = ["fort.218", "fort.222", "fort.224"]

    # file = "fort.217"
    for file in pres_files + wind_files:
        file1 = os.path.join(KAT_EX_PATH, file)
        file2 = os.path.join(DATA_PATH, "mult1", file)

        with open(file1) as file_1:
            file_1_text = file_1.readlines()

        with open(file2) as file_2:
            file_2_text = file_2.readlines()

        # Find and print the diff:
        for line in difflib.unified_diff(
            file_1_text, file_2_text, fromfile=file1, tofile=file2, lineterm=""
        ):
            print(line)


class Holland80:
    def __init__(self, pc, rmax, vmax) -> None:
        """ """
        self.pc = pc  # Pa
        self.rho = 1.0  # 15  # kg m-3
        self.pn = millibar_to_pascal(1010)  # Pa
        self.rmax = rmax  # meters
        self.vmax = vmax  # meters per second
        self.b_coeff = self.vmax**2 / (self.pn - self.pc) * np.e  # dimensionless

    def pressure(self, radii: np.ndarray) -> np.ndarray:
        return self.pn - (self.pn - self.pc) * np.exp(
            -((radii / self.rmax) ** self.b_coeff)
        )

    def velocity(self, radii: np.ndarray) -> float:
        return np.sqrt(
            (self.pn - self.pc)
            * np.exp(-((radii / self.rmax) ** self.b_coeff))
            * (radii / self.rmax) ** self.b_coeff
            * self.b_coeff
        )


class Holland08:
    def __init__(
        self,
        pc: float = 92800,
        rmax: float = 40744,
        vmax: float = 54.01667,
        xn: float = 1.1249,
    ) -> None:
        self.pc = pc  # Pa
        self.xn = xn
        self.rho = 1.0  # 15  # kg m-3
        self.pn = millibar_to_pascal(1010)  # Pa
        self.r64 = 2e5  # meters unused currently
        self.rmax = rmax  # meters
        self.vmax = vmax  # meters per second
        vf, pf = h08_vp(rmax, vmax, pc, self.pn, self.r64, self.rho, self.xn)
        self.vf = vf
        self.pf = pf

    def pressure(self, radii: np.ndarray) -> np.ndarray:
        return self.pf(radii)

    def velocity(self, radii: np.ndarray) -> float:
        return self.vf(radii)


def vmax_from_pressure_emanuel(
    pc: float, pn: float = millibar_to_pascal(1010)
) -> float:
    """
    Vmax from pressures using Emanuel 1988 relationships.

    V_{\max }=\sqrt{2 R_d T_s \ln \left(\frac{p_{\max }}{p_c}\right)},

    Args:
        pc (float): vmax.
        pn (float, optional): Defaults to millibar_to_pascal(1010).

    Returns:
        float:
    """
    from scipy.constants import R

    temp = 273.15 + 30
    # pmax = (pn - pc) * 1 / 10 + pc
    coeff = 54 / 20.66  # additional coeff added to fit Katrina
    # changed form pmax to pc
    return coeff * np.sqrt(2 * R * temp * np.log(pn / pc))


def vmax_from_pressure_holliday(
    pc: float, pn: float = millibar_to_pascal(1010)
) -> float:
    """
    Vmax from pressures using Atkinson Holliday 1997

    \mathrm{V}_{\max }=3.4(1010-\mathrm{MSLP})^{0.644}

    Args:
        pc (float): vmax.
        pn (float, optional): Defaults to millibar_to_pascal(1010).

    Returns:
        float: vmax
    """
    coeff = 54.01667 / 58.07310377789465  # change coeff to make it match Katrina
    return coeff * 3.4 * pascal_to_millibar(pn - pc) ** 0.644


def pmin_from_vmax(
    vmax: float, size: float, pn: float = millibar_to_pascal(1010)
) -> float:
    """
    \begin{array}{l}
    M S L P=23.286-0.483 V_{s r m}-\left(\frac{V_{s r m}}{24.254}\right)^2-12.587 S-0.483 \varphi
    +P_{\text {env }}
    \end{array}

    Returns:
        float: _description_
    """
    phi = NEW_ORLEANS.lat
    return (
        23.286 - 0.483 * vmax - (vmax / 24.254) ** 2 - 12.587 * size - 0.483 * phi + pn
    )


def _quad(a: float, b: float, c: float) -> Tuple[float]:
    return (
        (-b + np.sqrt(b**2 - 4 * a * c)) / 2 / a,
        (-b - np.sqrt(b**2 - 4 * a * c)) / 2 / a,
    )


def vmax_from_pressure_choi(
    pc: float, rmax: float = 40744, pn: float = millibar_to_pascal(1010)
):
    phi = NEW_ORLEANS.lat
    pc = pascal_to_millibar(pc)
    pn = pascal_to_millibar(pn)
    size = rmax / 10e3
    vmax = _quad(
        -1 / (24.254) ** 2, -0.483, pn - pc - 0.483 * phi - 12.587 * size + 23.286
    )
    return vmax


class ImpactSymmetricTC:
    def __init__(
        self,
        # vmax: float = 54.01667,  # m s**-1
        # rmax: float = 40744,  #  m
        # pc: float = 92800,  # Pa
        # pn: float = 100500,  # pa
        angle: float = 0.0,  # degrees
        trans_speed: float = 7.71,  # m s**-1,
        point: Point = NEW_ORLEANS,
        output_direc: str = os.path.join(DATA_PATH, "kat_h80"),  # string.
        symetric_model: any = Holland08(),
        debug: bool = False,
    ) -> None:
        """
        Generate Holland Hurricane Model.

        Args:
            vmax (float, optional): vmax. Defaults to 54.01667.
            point (Point, optional): point to hit. Defaults to NEW_ORLEANS.
            output_direc (str, optional): Output directory. Defaults to os.path.join(DATA_PATH, "kat_h80").
            symetric_model (any, optional): Symetric model. Defaults to Holland08().
            debug (bool, optional): Debug. Defaults to False.

        """
        self.angle = angle  # degrees
        self.trans_speed = trans_speed  # m s**-1
        print("Intializing", output_direc)
        self.output_direc = output_direc  # string to output direc
        # impact time for katrina.
        self.point = point
        self.impact_time = datetime.datetime(year=2005, month=8, day=29, hour=12)
        self.symetric_model = symetric_model
        self.debug = debug  # bool

    def center_from_time(self, time: np.datetime64) -> Point:
        """
        Assumes 111km per degree.

        # TODO: These centers need to be recorded in a netcdf file.

        Args:
            time (numpy.datetime64): time.

        Returns:
            Point: lon, lat.
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
            """
            Run ADICRC.

            Returns:
                int: output of os system.
            """
            print("Running ADCIRC", self.output_direc)
            command = f"cd {self.output_direc} \n {ADCIRC_EXE} > adcirc_log.txt"
            # Run ADCIRC in terminal
            return os.system(command)

        create_inputs()
        # quit if ADCIRC run fails.
        assert run_adcirc() == 0
        # output, error = process.communicate()
        # print(output, error)

    @timeit
    def prepare_run(self, forts: Tuple[str], timestep_smearing=True) -> None:
        """
        Prepare run.

        Args:
            forts (Tuple[str]): e.g. ("fort.221", "fort.222")
            timestep_smearing (bool, optional): Smear time to account for
            discretisation. Defaults to True.
        """
        da = read_owi_pressures(os.path.join(KAT_EX_PATH, forts[0]))
        average_timestep = (da.time.values[1:] - da.time.values[:-1]).mean()
        # smearing?
        vds_list = []  # velocity dataset list
        pds_list = []  # pressure dataset list
        lon_list = []  # list of storm centers. lon, lat
        lat_list = []

        for time in da.time.values:
            # work out time step for smearing
            # currently 3 hours -> subtract 1 hour + add 1 hour, then average?
            # This is where the time smearing happens.
            point = self.center_from_time(time)
            lon_list.append(point.lon)
            lat_list.append(point.lat)
            # TODO: make this more general.
            if timestep_smearing:
                vds1, pds1 = self.tc_time_slice(da, time - average_timestep / 3)
                vds2, pds2 = self.tc_time_slice(da, time)
                vds3, pds3 = self.tc_time_slice(da, time + average_timestep / 3)
                vds = xr.merge([vds1, vds2, vds3]).mean("time").expand_dims(dim="time")
                vds = vds.assign_coords({"time": [time]})
                pds = xr.merge([pds1, pds2, pds3]).mean("time").expand_dims(dim="time")
                pds = pds.assign_coords({"time": [time]})
            else:
                vds, pds = self.tc_time_slice(da, time)

            vds_list.append(vds)
            pds_list.append(pds)

        # print(type(vds_list))
        # print(vds_list[0])
        vds = xr.concat(vds_list, dim="time")
        pda = xr.concat(pds_list, dim="time")["pressure"]
        cda = xr.Dataset(
            data_vars=dict(lon=(["time"], lon_list), lat=(["time"], lat_list)),
            coords={"time": vds.time.values},
        )
        cda["lat"].attrs["units"] = "degrees_north"
        cda["lon"].attrs["units"] = "degrees_east"
        # print(cda)
        if self.debug:
            pda.to_netcdf(os.path.join(self.output_direc, forts[0]) + ".nc")
            vds.to_netcdf(os.path.join(self.output_direc, forts[1]) + ".nc")
        cda.to_netcdf(os.path.join(self.output_direc, "traj") + ".nc")
        write_owi_pressures(pda, os.path.join(self.output_direc, forts[0]))
        write_owi_windspeeds(vds, os.path.join(self.output_direc, forts[1]))

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


@timeit
def run_katrina_holland() -> None:
    """Run the Katrina as Holland 1980."""

    point = Point(NEW_ORLEANS.lon + 1.5, NEW_ORLEANS.lat)
    ImpactSymmetricTC(
        point=point, output_direc=os.path.join(DATA_PATH, "katd_h80")
    ).run_impact()


def run_katrina_h08() -> None:
    """Run the Katrina as Holland 2008."""

    point = Point(NEW_ORLEANS.lon + 0.4715, NEW_ORLEANS.lat)
    ImpactSymmetricTC(
        point=point,
        output_direc=os.path.join(DATA_PATH, "katf_h08"),
        symetric_model=Holland08(),
        debug=True,
    ).run_impact()


def point(x_diff: float) -> None:
    """Run the Katrina as Holland 2008."""

    point = Point(NEW_ORLEANS.lon + x_diff, NEW_ORLEANS.lat)
    folder = os.path.join(DATA_PATH, "kat_move_smeared")
    if not os.path.exists(folder):
        os.mkdir(folder)
    ImpactSymmetricTC(
        point=point,
        output_direc=os.path.join(folder, "x{:.3f}".format(x_diff) + "_kat_move"),
        symetric_model=Holland08(),
    ).run_impact()


def points() -> None:
    for x in np.linspace(-1, 2, num=100):
        point(x)


def angle_run(angle: float) -> None:
    """Run the Katrina as Holland 2008."""

    point = NEW_ORLEANS
    folder = os.path.join(DATA_PATH, "kat_angle")
    if not os.path.exists(folder):
        os.mkdir(folder)
    ImpactSymmetricTC(
        point=point,
        angle=angle,
        output_direc=os.path.join(folder, "a{:.3f}".format(angle) + "_kat_angle"),
        symetric_model=Holland08(),
    ).run_impact()


def angles() -> None:
    for angle in np.linspace(-90, 90, num=100):
        angle_run(angle)


def cangle_run(angle: float, prefix="c", lon_diff=1.2) -> None:
    """Run the Katrina as Holland 2008."""

    point = Point(NEW_ORLEANS.lon + lon_diff, NEW_ORLEANS.lat)
    folder = os.path.join(DATA_PATH, "kat_angle")
    if not os.path.exists(folder):
        os.mkdir(folder)
    ImpactSymmetricTC(
        point=point,
        angle=angle,
        output_direc=os.path.join(
            folder, prefix + "{:.3f}".format(angle) + "_kat_angle"
        ),
        symetric_model=Holland08(),
    ).run_impact()


def cangles() -> None:
    for angle in np.linspace(-90, 90, num=100):
        cangle_run(angle, prefix="e", lon_diff=-0.6)


def landfall_speed(speed: float, prefix="c", lon_diff=1.2) -> None:
    """Run the Katrina as Holland 2008."""
    point = Point(NEW_ORLEANS.lon + lon_diff, NEW_ORLEANS.lat)
    folder = os.path.join(DATA_PATH, "kat_landfall")
    if not os.path.exists(folder):
        os.mkdir(folder)
    ImpactSymmetricTC(
        point=point,
        trans_speed=speed,
        output_direc=os.path.join(
            folder, prefix + "{:.3f}".format(speed) + "_kat_landfall"
        ),
        symetric_model=Holland08(),
    ).run_impact()


def speeds() -> None:
    for speed in np.linspace(3, 12, num=20):
        landfall_speed(speed, prefix="c", lon_diff=1.2)


def pc_holliday(pc: float, prefix="c", lon_diff=1.2) -> None:
    vmax = vmax_from_pressure_holliday(pc)
    print(prefix, pc, vmax)
    point = Point(NEW_ORLEANS.lon + lon_diff, NEW_ORLEANS.lat)
    folder = os.path.join(DATA_PATH, "kat_pcf")
    if not os.path.exists(folder):
        os.mkdir(folder)
    ImpactSymmetricTC(
        point=point,
        output_direc=os.path.join(folder, prefix + "{:.3f}".format(pc) + "_kat_pc"),
        symetric_model=Holland08(pc=pc, vmax=vmax),
    ).run_impact()


def pcs() -> None:
    for pc in millibar_to_pascal(np.linspace(900, 980, num=100)):
        pc_holliday(pc, prefix="a", lon_diff=0.0)


def rmax_vary(rmax: float, prefix="c", lon_diff=1.2) -> None:
    print(prefix, rmax)
    point = Point(NEW_ORLEANS.lon + lon_diff, NEW_ORLEANS.lat)
    folder = os.path.join(DATA_PATH, "kat_rmax")
    if not os.path.exists(folder):
        os.mkdir(folder)
    ImpactSymmetricTC(
        point=point,
        output_direc=os.path.join(folder, prefix + "{:.3f}".format(rmax) + "_kat_rmax"),
        symetric_model=Holland08(rmax=rmax),
    ).run_impact()


def rmaxs() -> None:
    for rmax in nmile_to_meter(np.linspace(10, 40, num=40)):
        rmax_vary(rmax, prefix="a", lon_diff=0.0)


def xn_vary(xn: float, prefix="c", lon_diff=1.2) -> None:
    print(prefix, xn)
    point = Point(NEW_ORLEANS.lon + lon_diff, NEW_ORLEANS.lat)
    folder = os.path.join(DATA_PATH, "kat_xn")
    if not os.path.exists(folder):
        os.mkdir(folder)
    ImpactSymmetricTC(
        point=point,
        output_direc=os.path.join(folder, prefix + "{:.3f}".format(xn) + "_kat_xn"),
        symetric_model=Holland08(xn=xn),
    ).run_impact()


def xns() -> None:
    for xn in np.linspace(0.3, 2, num=40):
        # xn_vary(xn, prefix="a", lon_diff=0.0)
        # xn_vary(xn, prefix="b", lon_diff=0.6)
        xn_vary(xn, prefix="c", lon_diff=1.2)


if __name__ == "__main__":
    # for key in tc.MODEL_VANG:
    #    plot_katrina_windfield_example(model=key)
    # plot_katrina_windfield_example(model="H08")
    # python src/models/generation.py
    # print(NEW_ORLEANS)
    # mult_generation(1)
    # [mult_generation(x / 4) for x in range(16) if x not in list(range(0, 16, 4))]
    # comp()
    run_katrina_h08()
    # cangles()
    # run_katrina_h08()  # speeds()
    # rmaxs()
    # xns()
    # points()
    # print(vmax_from_pressure_holliday(92800))
    # print(vmax_from_pressure_emanuel(92800))
    # print(vmax_from_pressure_choi(92800))
    # run_katrina_h08()
    # print("ok")
    # output_direc = os.path.join(DATA_PATH, "mult2")
    # adcirc_exe = "/Users/simon/adcirc-swan/adcircpy/exe/adcirc"
    # command = f"cd {output_direc} \n {adcirc_exe} > adcirc_log.txt"
    # os.system(command)
    # original
    # iLat= 100iLong= 100DX=0.2500DY=0.2500SWLat=17.00000SWLon=-99.0000DT=200508250000
    # new
    # iLat= 100iLong= 100DX=0.2500DY=0.2500SWLat=17.00000SWLon=-99.0000DT=200508250000
