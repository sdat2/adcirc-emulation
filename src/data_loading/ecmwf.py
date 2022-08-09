"""Get ERA5 by CDS API calls."""
from typing import List, Union
from datetime import datetime, date, timedelta
import numpy as np
import cdsapi
from src.constants import GOM_BBOX, KATRINA_ERA5_NC, ECMWF_AIR_VAR, ECWMF_WATER_VAR

DATEFORMAT = "%Y-%m-%d"
HOURS = [
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
]
MONTHS = [
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
]


def str_to_date(strdate: Union[str, any], dateformat: str = DATEFORMAT) -> any:
    """
    String to date.

    Args:
        strdate (Union[str, any]): The string date encoded by the dateformat.
        dateformat (str, optional): The format of the date. Defaults to DATEFORMAT.

    Returns:
        any: Datetime object.

    Example:
        >>> from src.data_loading.ecmwf import str_to_date
        >>> date = str_to_date("2005-08-25")
        >>> date.year
        2005
        >>> date.month
        8
        >>> date.day
        25
    """
    if isinstance(strdate, str):
        strdate = datetime.strptime(strdate, dateformat)
    return strdate


def two_char_int(int_input: int) -> str:
    ret_str = str(int_input)
    if len(ret_str) == 1:
        ret_str = "0" + ret_str
    return ret_str


def date_to_str(date: any) -> str:
    month_str = two_char_int(date.month)
    day_str = two_char_int(date.day)
    return str(date.year) + "-" + month_str + "-" + day_str


def end_of_year(year) -> str:
    return str(year) + "-" + str(12) + "-" + str(31)


def start_of_year(year) -> str:
    return str(year) + "-" + "01" + "-" + "01"


def end_of_month(date_inp):
    if date_inp.month != 12:
        return date_to_str(
            date(int(date_inp.year), int(date_inp.month + 1), 1) - timedelta(days=1)
        )
    else:
        return end_of_year(date_inp.year)


def start_of_month(date_inp):
    month_str = two_char_int(date_inp.month)
    return str(date_inp.year) + "-" + month_str + "-" + "01"


def year_month_day_lists(
    startdate: Union[np.datetime64, str], enddate: Union[np.datetime64, str]
) -> List[List[str]]:
    """
    Month Day lists for running cds api.

    !Inclusive counting!

    if str formatted as '%Y-%m-%d'

    Args:
        startdate (np.datetime64): Start date.
        enddate (np.datetime64): End date.

    Returns:
        List[List[str]]: [[Year, Month, [Day1, Day2, ...]], [[Month2], ...]]

    Examples of use::
        >>> from src.data_loading.ecmwf import year_month_day_lists
        >>> year_month_day_lists("2005-08-20", "2005-08-31")
            [['2005', '08', ['20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31']]]
        >>> year_month_day_lists("2021-08-29", "2021-09-05")
            [['2021', '08', ['29', '30', '31']], ['2021', '09', ['01', '02', '03', '04', '05']]]
    """
    startdate = str_to_date(startdate)
    enddate = str_to_date(enddate)

    assert startdate < enddate

    if startdate.year < enddate.year:
        start_part = year_month_day_lists(startdate, end_of_year(startdate.year))
        end_part = year_month_day_lists(start_of_year(enddate.year), enddate)
        if startdate.year < enddate.year - 1:
            intermediate_piece = []
            for year in range(startdate.year + 1, enddate.year):
                intermediate_piece += year_month_day_lists(
                    start_of_year(year), end_of_year(year)
                )
            final_list = start_part + intermediate_piece + end_part
        else:
            final_list = start_part + end_part
    elif startdate.month < enddate.month:
        start_part = year_month_day_lists(startdate, end_of_month(startdate))
        end_part = year_month_day_lists(start_of_month(enddate), enddate)
        if startdate.month < enddate.month - 1:
            intermediate_piece = []
            for month in range(startdate.month + 1, enddate.month):
                temp_date = date(startdate.year, month, 1)
                intermediate_piece += year_month_day_lists(
                    start_of_month(temp_date), end_of_month(temp_date)
                )
            final_list = start_part + intermediate_piece + end_part
        else:
            final_list = start_part + end_part
    else:
        final_list = [
            [
                str(startdate.year),
                two_char_int(startdate.month),
                [two_char_int(x) for x in range(startdate.day, enddate.day + 1)],
            ]
        ]

    return final_list


def katrina_era5(var=ECMWF_AIR_VAR + ECWMF_WATER_VAR) -> None:
    """
    Get Katrina ERA5.

    """

    date_list = year_month_day_lists("2005-08-20", "2005-08-31")

    client = cdsapi.Client()
    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": var,
            "year": date_list[0][0],
            "month": date_list[0][1],
            "day": date_list[0][2],
            "time": HOURS,
            "area": GOM_BBOX.ecmwf(),
        },
        KATRINA_ERA5_NC,
    )


def monthly_avgs(var=ECMWF_AIR_VAR + ECWMF_WATER_VAR) -> None:
    """
    Make all the monthly average netctdfs for rthe full time period.
    """
    client = cdsapi.Client()
    for var in var:
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
                "month": MONTHS,
                "area": GOM_BBOX.ecmwf(),
            },
            var + ".nc",
        )


if __name__ == "__main__":
    # python src/data_loading/ecmwf.py
    print(year_month_day_lists("2003-08-11", "2006-01-02"))
