# Place all your constants here
import os

# Note: constants should be UPPER_CASE
constants_path = os.path.realpath(__file__)
SRC_PATH = os.path.dirname(constants_path)
PROJECT_PATH = os.path.dirname(SRC_PATH)
DATA_PATH = os.path.join(PROJECT_PATH, "data")

NEW_ORLEANS = [-90.0715, 29.9511]  # lon , lat
DEFAULT_GUAGES = ["8761724", "8761927", "8761955", "8762075", "8762482"]
KATRINA_TIDE_NC = os.path.join(DATA_PATH, "katrina_tides.nc")
