"""Get ERA5 by CDS API calls."""
from typing import List
import numpy as np
import cdsapi
from src.constants import GOM, KATRINA_ERA5_NC


def month_day_lists(
    startdate: np.datetime64, enddate: np.datetime64
) -> List[List[str], List[str]]:
    """
    Month Day lists for running cds api.

    Not yet implemented.

    Args:
        startdate (np.datetime64): Start date.
        enddate (np.datetime64): End date.

    Returns:
        List[List[str], List[str]]: [[Month], [Day1, Day2, ...], ...]
    """
    return [
        [
            ["08"],
            ["20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31",],
        ]
    ]


def katrina_era5() -> None:
    """
    Get Katrina ERA5.

    # TODO: Add the ability to straddle months.
    """
    c = cdsapi.Client()
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": [
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "2m_dewpoint_temperature",
                "2m_temperature",
                "mean_sea_level_pressure",
                "mean_wave_direction",
                "mean_wave_period",
                "sea_surface_temperature",
                "significant_height_of_combined_wind_waves_and_swell",
                "surface_pressure",
                "total_precipitation",
            ],
            "year": "2005",
            "month": "08",
            "day": [
                "20",
                "21",
                "22",
                "23",
                "24",
                "25",
                "26",
                "27",
                "28",
                "29",
                "30",
                "31",
            ],
            "time": [
                "00:00",
                "01:00",
                "02:00",
                "03:00",
                "04:00",
                "05:00",
                "06:00",
                "07:00",
                "08:00",
                "09:00",
                "10:00",
                "11:00",
                "12:00",
                "13:00",
                "14:00",
                "15:00",
                "16:00",
                "17:00",
                "18:00",
                "19:00",
                "20:00",
                "21:00",
                "22:00",
                "23:00",
            ],
            "area": GOM,
        },
        KATRINA_ERA5_NC,
    )


if __name__ == "__main__":
    # python src/data_loading/ecmwf.py
    katrina_era5()
