"""Tidal Comparison Plots."""
import os
import xarray as xr
import datetime
import matplotlib.pyplot as plt
from sithom.plot import plot_defaults
from src.data_loading.tides import filtered_tidal_gauges
from src.data_loading.adcirc import select_coastal_cells
from src.constants import KAT_EX_PATH, FIGURE_PATH


def tide_plot(stationid=0):
    # python src/plot/tides.py
    tds = filtered_tidal_gauges()
    print(tds)
    psc = tds.isel(stationid=stationid)
    print((float(psc.lon.values), float(psc.lat.values)))

    lons, lats, heights = select_coastal_cells(
        float(psc.lon.values), float(psc.lat.values)
    )
    start = datetime.datetime(year=2005, month=8, day=19, hour=5)
    time_step = datetime.timedelta(hours=1, minutes=20)
    # start = datetime.datetime(year=2005, month=8, day=21, hour=18)
    # time_step = datetime.timedelta(hours=1)
    # time_step = datetime.timedelta(hours=1, minutes=15)
    # start = datetime.datetime(year=2005, month=8, day=20, hour=18)

    print(heights.shape)
    ds = xr.Dataset(
        data_vars=dict(height=(["time", "point"], heights)),
        coords=dict(
            lon=(["point"], lons),
            lat=(["point"], lats),
            time=[start + i * time_step for i in range(heights.shape[0])],
        ),
    )
    print(ds)
    plot_defaults()

    psc.water_level.plot()
    ds.height.plot.line(hue="point", alpha=0.5)
    plt.savefig(os.path.join(FIGURE_PATH, "tide_gauge" + str(stationid) + ".png"))
    plt.clf()


if __name__ == "__main__":
    # python src/plot/tides.py
    [tide_plot(x) for x in range(4)]
