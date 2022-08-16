"""Generate hurricane."""
from typing import Tuple, Literal
import numpy as np
import matplotlib.pyplot as plt
import climada.hazard.trop_cyclone as tc
from src.constants import NO_BBOX
from src.data_loading.ibtracs import katrina, prep_for_climada

MODEL_VANG = {'H08': 0, 'H1980': 1, 'H10': 2}


def make_katrina_windfields(model: str) -> Tuple[np.ndarray]:
    """
    Make Katrina

    Args:
        model (str): e.g. H08

    Returns:
        Tuple[np.ndarray]: _description_
    """
    centroids = np.array(
        [
            [x, y]
            for x in np.linspace(*NO_BBOX.lat, num=50)
            for y in np.linspace(*NO_BBOX.lon, num=50)
        ]
    )
    return tc.compute_windfields(
        prep_for_climada(katrina()), centroids, MODEL_VANG[model], #metric="equirect"
    )


def plot_katrina_windfield_example() -> None:
    output = make_katrina_windfields()
    _, axs = plt.subplots(2, 1, sharex=True)
    axs[0].plot(output[0][50, :, 0])
    axs[0].set_ylabel("Windfield [x]")
    axs[1].plot(output[0][50, :, 1])
    axs[1].set_ylabel("Windfield [y]")
    axs[1].set_xlabel("Node")

if __name__ == "__main__":
    # python src/models/generate-hurricane.py
    print("ok")
    good()
