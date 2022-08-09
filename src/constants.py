# Place all your constants here
import os
from typing import List
from src.place import BoundingBox, Point

# Note: constants should be UPPER_CASE
constants_path = os.path.realpath(__file__)
SRC_PATH = os.path.dirname(constants_path)
PROJECT_PATH = os.path.dirname(SRC_PATH)
DATA_PATH = os.path.join(PROJECT_PATH, "data")

REPORT_PATH = os.path.join(PROJECT_PATH, "report")
FIGURE_PATH = os.path.join(REPORT_PATH, "figures")

DEFAULT_GAUGES = [
    "8729840",
    "8735180",
    "8760922",
    "8761724",
    "8762075",
    "8762482",
    "8764044",
]

# Data files.
KATRINA_TIDE_NC = os.path.join(DATA_PATH, "katrina_tides.nc")
KATRINA_ERA5_NC = os.path.join(DATA_PATH, "katrina_era5.nc")
IBTRACS_NC = os.path.join(DATA_PATH, "IBTrACS.ALL.v04r00.nc")
MID_KATRINA_TIME: str = "2005-08-29T10:00:00"

# regional bounding boxes for ERA5 download.
# Gulf of Mexico
# lat+, lon-, lat-, lon+
GOM = [35, -100, 15, -80]
GOM_BBOX = BoundingBox([-100, -80], [15, 35])
NO_BBOX = BoundingBox([-92, -86.5], [28.5, 30.8])  # zoomed in around New orleans
# Significant places
NEW_ORLEANS = Point(-90.0715, 29.9511)  # lon , lats

ECMWF_AIR_VAR: List[str] = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_dewpoint_temperature",
    "2m_temperature",
    "mean_sea_level_pressure",
    "surface_pressure",
    "total_precipitation",
]

ECMWF_WATER_VAR: List[str] = [
    "mean_wave_direction",
    "mean_wave_period",
    "sea_surface_temperature",
    "significant_height_of_combined_wind_waves_and_swell",
]
