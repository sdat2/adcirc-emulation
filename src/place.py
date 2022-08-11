"""Place objects."""
from typing import List
import matplotlib.axes
from typeguard import typechecked

# import matplotlib.pyplot as plt


@typechecked
class BoundingBox:
    """
    BoundingBox class to deal with the varying output requirments often needed to
    describe the same geographical box to different APIs.

    Example::
        >>> from src.place import BoundingBox
        >>> bbox = BoundingBox([-30, 30], [10, 30], desc="")
        >>> bbox.cartopy() # [lat-, lat+, lon-, lon+]
        [-30, 30, 10, 30]
        >>> bbox.ecmwf() # [lat+, lon-, lat-, lon+]
        [30, -30, 10, 30]
    """

    def __init__(self, lon: List[float], lat: List[float], desc: str = "NONE") -> None:
        """
        Create BBOX.

        Args:
            lon (List[float]): Longitude bounds. Degrees East.
            lat (List[float]): Latitude bounds. Degrees North.
            desc (str): Description of boundary box for debugging. Defaults to "None".
        """
        assert len(lon) == 2
        assert len(lat) == 2
        self.lon = lon
        self.lat = lat
        self.desc = desc

    def __repr__(self) -> str:
        """
        Representation string.
        """
        return [
            ("Latitude bounds", self.lat, "degrees_north"),
            ("Longitude bounds", self.lon, "degrees_east"),
            self.desc,
        ]

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


@typechecked
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

    def bbox(self, buffer: float = 3) -> BoundingBox:
        """
        Get bbox padding a central location.

        Size of the square is 4 * buffer**2.

        Args:
            loc (Point): Defaults to NEW_ORLEANS.
            buffer (float, optional): How many degrees to go out from loc. Defaults to 1.

        Returns:
            BoundingBox: A bounding box like [-91.0715, 28.9511, -89.0715, 30.9511].

        Example::
            >>> from src.place import Point
            >>> point = Point(20, 30)
            >>> bbox = point.bbox(2)
            >>> bbox.cartopy()
            [18, 22, 28, 32]

        """
        return BoundingBox(
            [self.lon - buffer, self.lon + buffer],
            [self.lat - buffer, self.lat + buffer],
        )
