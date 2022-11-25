"""Tidal Comparison Plots."""
import xarray as xr
import datetime
import matplotlib.pyplot as plt
from sithom.plot import plot_defaults
from src.data_loading.tides import filtered_tidal_gauges
from src.data_loading.adcirc import select_coastal_cells


if __name__ == "__main__":
    # python src/plot/tides.py
    tds = filtered_tidal_gauges()
    print(tds)
    psc = tds.isel(stationid=0)
    print((float(psc.lon.values), float(psc.lat.values)))

    lons, lats, heights = select_coastal_cells(
        float(psc.lon.values), float(psc.lat.values)
    )
    start = datetime.datetime(year=2005, month=8, day=19, hour=7)
    time_step = datetime.timedelta(hours=1.33)

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
    # plot_defaults()

    psc.water_level.plot()
    ds.height.plot.line(hue="point", alpha=0.5)
    plt.show()
