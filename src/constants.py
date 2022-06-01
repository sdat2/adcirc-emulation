# Place all your constants here
import os

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
DEFAULT_GUAGES = [
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
