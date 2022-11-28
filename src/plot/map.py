"""Function to plot maps using cartopy."""
import matplotlib
import matplotlib.pyplot as plt
from typeguard import typechecked

try:
    import cartopy
    import cartopy.crs as ccrs
except ImportError:
    print("cartopy not installed")


@typechecked
def add_features(ax: matplotlib.axes.Axes) -> matplotlib.axes.Axes:
    """
    Add features to map.

    Args:
        ax (cartopy.mpl.geoaxes.GeoAxesSubplot): Axes.

    Returns:
        cartopy.mpl.geoaxes.GeoAxesSubplot: Cartopy axes.
    """
    ax.add_feature(cartopy.feature.COASTLINE, alpha=0.5)
    ax.add_feature(cartopy.feature.LAKES, alpha=0.5)
    ax.add_feature(cartopy.feature.BORDERS, linestyle=":")
    ax.add_feature(cartopy.feature.RIVERS)
    return ax


@typechecked
def map_axes() -> matplotlib.axes.Axes:
    """
    Map axes with features.

    Returns:
        cartopy.mpl.geoaxes.GeoAxesSubplot: Map axes with features.
    """
    return add_features(plt.axes(projection=ccrs.PlateCarree()))


if __name__ == "__main__":
    # python src/plot/map.py
    print(type(map_axes()))
