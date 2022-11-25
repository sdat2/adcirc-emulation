"""Tidal Comparison Plots."""
from src.data_loading.tides import filtered_tidal_gauges
from src.data_loading.adcirc import select_coastal_cells


if __name__ == "__main__":
    # python src/plot/tides.py
    tds = filtered_tidal_gauges()
    print(tds)
    psc = tds.isel(stationid=0)
    print((float(psc.lon.values), float(psc.lat.values)))
    print(select_coastal_cells(float(psc.lon.values), float(psc.lat.values)))
