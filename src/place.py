"""Place objects."""
from typing import List
import matplotlib
from typeguard import typechecked

# import matplotlib.pyplot as plt


class BoundingBox:
    """
    BoundingBox class to deal with the varying output requirments often needed to
    describe the same geographical box to different APIs.
    """

    def __init__(self, lon: List[float], lat: List[float], desc: str = "NONE") -> None:
        """
        Create BBOX.

        Args:
            lon (List[float]): Degrees East.
            lat (List[float]): Degrees North.
        """
        assert len(lon) == 2
        assert len(lat) == 2
        self.lon = lon
        self.lat = lat
        self.desc = desc
        print(lon, lat)

    def __repr__(self) -> str:
        """
        Representation string.
        """
        return str(self.ecmwf())

    def cartopy(self) -> List[float]:
        """
        Cartopy style bounding box.

        Returns:
            List[float]: [lon-, lon+, lat-, lat+]
        """
        return self.lon + self.lat

    def ecmwf(self) -> List[float]:
        """
        ECMWF style bounding box.

        Returns:
            List[float]: [lat+, lon-, lat-, lon+]
        """
        return [self.lat[1], self.lon[0], self.lat[0], self.lon[1]]

    def ax_lim(self, ax: matplotlib.axes.Axes):
        """
        Apply ax limit to graph.

        Args:
            ax (matplotlib.axes.Axes): _description_
        """
        ax.set_xlim(self.lon)
        ax.set_ylim(self.lat)


class Point:
    def __init__(self, lon: float, lat: float) -> None:
        """
        Initialise point.

        Args:
            lon (float): Longitude.
            lat (float): Latitude.
        """
        self.lon = lon
        self.lat = lat


def bbox_from_point(point: Point, buffer: float = 3) -> BoundingBox:
    """
    Get bbox padding a central location.

    Size of the square is 4 * buffer**2.

    Args:
        loc (Point): Defaults to NEW_ORLEANS.
        buffer (float, optional): How many degrees to go out from loc. Defaults to 1.

    Returns:
        BoundingBox: A bounding box like [-91.0715, 28.9511, -89.0715, 30.9511].
    """
    return BoundingBox(
        [point.lon - buffer, point + buffer], [point.lat - buffer, point.lat + buffer]
    )
