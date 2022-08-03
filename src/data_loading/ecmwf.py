"""Get ERA5 by CDS API calls."""
from typing import List
import numpy as np
import cdsapi
from src.constants import GOM_BBOX, KATRINA_ERA5_NC


def month_day_lists(
    startdate: np.datetime64, enddate: np.datetime64
) -> List[List[str]]:
    """
    Month Day lists for running cds api.

    Not yet implemented.

    Args:
        startdate (np.datetime64): Start date.
        enddate (np.datetime64): End date.

    Returns:
        List[List[str]]: [[[Month], [Day1, Day2, ...]], [[Month2], ...]]
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

    """

    air_var = [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_dewpoint_temperature",
        "2m_temperature",
        "mean_sea_level_pressure",
        "surface_pressure",
        "total_precipitation",
    ]

    water_var = [
        "mean_wave_direction",
        "mean_wave_period",
        "sea_surface_temperature",
        "significant_height_of_combined_wind_waves_and_swell",
    ]

    client = cdsapi.Client()
    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": air_var + water_var,
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
            "area": GOM_BBOX.ecmwf(),
        },
        KATRINA_ERA5_NC,
    )


def monthly_avgs() -> None:
    """
    Make all the monthly average netctdfs for rthe full time period.
    """
    air_var = [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_dewpoint_temperature",
        "2m_temperature",
        "mean_sea_level_pressure",
        "surface_pressure",
        "total_precipitation",
    ]

    water_var = [
        "mean_wave_direction",
        "mean_wave_period",
        "sea_surface_temperature",
        "significant_height_of_combined_wind_waves_and_swell",
    ]

    client = cdsapi.Client()
    for var in air_var + water_var:
        client.retrieve(
            "reanalysis-era5-single-levels-monthly-means",
            {
                "format": "netcdf",
                "year": [
                    "1959",
                    "1960",
                    "1961",
                    "1962",
                    "1963",
                    "1964",
                    "1965",
                    "1966",
                    "1967",
                    "1968",
                    "1969",
                    "1970",
                    "1971",
                    "1972",
                    "1973",
                    "1974",
                    "1975",
                    "1976",
                    "1977",
                    "1978",
                    "1979",
                    "1980",
                    "1981",
                    "1982",
                    "1983",
                    "1984",
                    "1985",
                    "1986",
                    "1987",
                    "1988",
                    "1989",
                    "1990",
                    "1991",
                    "1992",
                    "1993",
                    "1994",
                    "1995",
                    "1996",
                    "1997",
                    "1998",
                    "1999",
                    "2000",
                    "2001",
                    "2002",
                    "2003",
                    "2004",
                    "2005",
                    "2006",
                    "2007",
                    "2008",
                    "2009",
                    "2010",
                    "2011",
                    "2012",
                    "2013",
                    "2014",
                    "2015",
                    "2016",
                    "2017",
                    "2018",
                    "2019",
                    "2020",
                    "2021",
                    "2022",
                ],
                "product_type": "monthly_averaged_reanalysis",
                "variable": [var],
                "time": "00:00",
                "month": [
                    "01",
                    "02",
                    "03",
                    "04",
                    "05",
                    "06",
                    "07",
                    "08",
                    "09",
                    "10",
                    "11",
                    "12",
                ],
                "area": GOM_BBOX.ecmwf(),
            },
            var + ".nc",
        )


if __name__ == "__main__":
    # python src/data_loading/ecmwf.py
    katrina_era5()
