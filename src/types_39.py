import numpy as np
import typing
from typing import TypeVar, Generic, Tuple, Union, Optional
from typeguard import typechecked

Shape = TypeVar("Shape")
DType = TypeVar("DType")


class Array(np.ndarray, Generic[Shape, DType]):
    """
    Use this to type-annotate numpy arrays, e.g.
        image: Array['H,W,3', np.uint8]
        xy_points: Array['N,2', float]
        nd_mask: Array['...', bool]
    """

    pass


@typechecked
def compute_l2_norm(arr: Array["N,2", float]) -> Array["N", float]:
    return (arr**2).sum(axis=1) ** 0.5

if __name__ == "__main__":
    print(compute_l2_norm(arr=np.array([(1, 2), (3, 1.5), (0, 5.5)])))

# typing
# scalene
