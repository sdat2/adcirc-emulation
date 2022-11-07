from __future__ import annotations
from typing import TypeVar, TYPE_CHECKING
import numpy as np
from typeguard import typechecked
import numpy.typing as npt

T1 = TypeVar("T1", bound=npt.NBitBase)
T2 = TypeVar("T2", bound=npt.NBitBase)


# @typechecked
def add(a: np.floating[T1], b: np.integer[T2]) -> np.floating[T1 | T2]:
    return a + b


a = np.float16()
b = np.int64()
out = add(a, b)

if TYPE_CHECKING:
    reveal_locals()
