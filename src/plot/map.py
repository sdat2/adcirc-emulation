"""Function to plot maps using cartopy."""
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs


def add_features(ax):
    """
    Add features to map.

    Args:
        ax (cartopy.mpl.geoaxes.GeoAxesSubplot): Axes.

    Returns:
        cartopy.mpl.geoaxes.GeoAxesSubplot: cartopy axes..
    """
    ax.add_feature(cartopy.feature.COASTLINE)
    ax.add_feature(cartopy.feature.LAKES, alpha=0.5)
    ax.add_feature(cartopy.feature.BORDERS, linestyle=":")
    ax.add_feature(cartopy.feature.RIVERS)
    return ax


def map_axes():
    """
    Map axes with features.

    Returns:
        cartopy.mpl.geoaxes.GeoAxesSubplot: Map axes with features.
    """
    return add_features(plt.axes(projection=ccrs.PlateCarree()))


if __name__ == "__main__":
    # python src/plot/map.py
    print(type(map_axes()))
