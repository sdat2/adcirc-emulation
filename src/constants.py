"""`src/constants.py`."""
# Place all your constants here
import os
from typing import List
from sithom.place import BoundingBox, Point
from pint import UnitRegistry

# Physical units.
UREG = UnitRegistry()  # Common unit registry between all modules.
RADIUS_EARTH = 6371.009 * UREG.kilometer

# Note: constants should be UPPER_CASE

# /home/users/sithom/new-orleans
constants_path: str = os.path.realpath(__file__)
SRC_PATH: str = os.path.dirname(constants_path)
CONFIG_PATH: str = os.path.join(SRC_PATH, "config")
PROJECT_PATH: str = os.path.dirname(SRC_PATH)
DATA_PATH: str = os.path.join(PROJECT_PATH, "data")
REPORT_PATH: str = os.path.join(PROJECT_PATH, "report")
FIGURE_PATH: str = os.path.join(REPORT_PATH, "no")
ROOT: str = os.path.dirname(PROJECT_PATH)
SECRET_PATH = os.path.join(ROOT, ".secret")
ADCIRC_EXE: str = os.path.join(ROOT, "adcirc-swan", "adcirc", "work", "adcirc")
# ADCIRC_EXE: str = "/Users/simon/adcirc-swan/adcircpy/exe/adcirc"

LANDING_DS: str = os.path.join(DATA_PATH, "IBTrACS_gom_landings.nc")

# Katrina example:
KAT_EX_PATH = os.path.join(
    ROOT, "adcirc-swan", "testsuite", "adcirc", "adcirc_katrina-2d"
)
# print(ADCIRC_EXE, KAT_EX_PATH)

# PATH=/Users/simon/adcirc-swan/adcircpy/exe:$PATH

# Default tidal gauges.
DEFAULT_GAUGES: List[str] = [
    "8729840",
    "8735180",
    "8760922",
    "8761724",
    "8762075",
    "8762482",
    "8764044",
]

# Default CRS
WGS84: str = "EPSG:4326"  # WGS84 standard crs (latitude, longitude)

# Data files.
KATRINA_TIDE_NC: str = os.path.join(DATA_PATH, "katrina_tides.nc")
KATRINA_ERA5_NC: str = os.path.join(DATA_PATH, "katrina_era5.nc")
KATRINA_WATER_ERA5_NC: str = os.path.join(DATA_PATH, "katrina_water_era5.nc")
IBTRACS_NC: str = os.path.join(DATA_PATH, "IBTrACS.ALL.v04r00.nc")
MID_KATRINA_TIME: str = "2005-08-29T10:00:00"

# regional bounding boxes for ERA5 download.
# Gulf of Mexico box (lons, lats)
GOM_BBOX = BoundingBox([-100, -80], [15, 35], desc="Gulf of Mexico Bounding Box")
# New Orleans box (lons, lats)
NO_BBOX = BoundingBox([-92, -86.5], [28.5, 30.8], desc="New Orleans Area Bounding Box")
# Significant places (lon, lat)
NEW_ORLEANS = Point(-90.0715, 29.9511, desc="New Orleans Point")  # lon , lats

# ERA5 atmospheric variables.
ECMWF_AIR_VAR: List[str] = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_dewpoint_temperature",
    "2m_temperature",
    "mean_sea_level_pressure",
    "surface_pressure",
    "total_precipitation",
    "sea_surface_temperature",
]

# ERA5 oceanic variables.
ECMWF_WATER_VAR: List[str] = [
    "mean_wave_direction",
    "mean_wave_period",
    # "sea_surface_temperature",
    "significant_height_of_combined_wind_waves_and_swell",
]

# dateformat string.
DATEFORMAT: str = "%Y-%m-%d"


# plotting defaults
FEATURE_LIST = ["angle", "speed", "point_east", "rmax", "pc", "xn"]
LABEL_LIST = [
    "Bearing, $\chi$",
    "Translation Speed, $v_t$",
    "Longitude Displacement, $c$",
    "Radius of Maximum Wind, $r_{\mathrm{max}}$",
    "Central Pressure, $p_c$",
    "Decay Exponent, $x_n$",
]
SYMBOL_LIST = ["$\chi$", "$v_t$", "$c$", "$r_{\mathrm{max}}$", "$p_c$", "$x_n$"]
LABEL_DICT = {FEATURE_LIST[i]: LABEL_LIST[i] for i in range(len(FEATURE_LIST))}
SYMBOL_DICT = {FEATURE_LIST[i]: SYMBOL_LIST[i] for i in range(len(FEATURE_LIST))}

PLACES_D = dict(
    ansley=27,
    new_orleans=5,
    diamondhead=17,
    mississippi=77,
    atchafayala=82,
    dulac=86,
    akers=2,
)
