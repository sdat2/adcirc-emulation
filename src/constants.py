# Place all your constants here
import os
from typing import List

# Note: constants should be UPPER_CASE
constants_path = os.path.realpath(__file__)
SRC_PATH = os.path.dirname(constants_path)
PROJECT_PATH = os.path.dirname(SRC_PATH)
DATA_PATH = os.path.join(PROJECT_PATH, "data")

REPORT_PATH = os.path.join(PROJECT_PATH, "report")
FIGURE_PATH = os.path.join(REPORT_PATH, "figures")

# Significant places
NEW_ORLEANS = [-90.0715, 29.9511]  # lon , lat

# regional bounding boxes for ERA5 download.
# Gulf of Mexico
# lat+, lon-, lat-, lon+
GOM = [35, -100, 15, -80]
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

ZOOMED_IN_LONS = [-92, -86.5]  # zoomed in around New orleans
ZOOMED_IN_LATS = [28.5, 30.8]
MID_KATRINA_TIME = "2005-08-29T10:00:00"

class BoundingBox():
    def __init__(self, lon: List[float], lat: List[float]) -> None:
        """
        Create BBOX.

        Args:
            lon (List[float]): Degrees East.
            lat (List[float]): Degrees North.
        """
        self.lon = lon
        self.lat = lat
        print(lon, lat)

    def __repr__(self) -> str:
        """
        Representation string.
        """
        return str(self.ecmwf())

    def cartopy(self) -> List[float]:
        """
        Cartopy style bounding box.

        Returns:
            List[float]: [lon-, lon+, lat-, lat+]
        """
        return self.lon + self.lat

    def ecmwf(self) -> List[float]:
        """
        ECMWF style bounding box.

        Returns:
            List[float]: ecmwf.
        """
        return [self.lat[1], self.lon[0], self.lat[0], self.lon[1]]


GOM_BBOX = BoundingBox([-100, -80], [15, 35])
NO_BBOX = BoundingBox([-92, -86.5], [28.5, 30.8])

