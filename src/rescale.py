"""Rescale the numbers to fall in 0.0 to 1.0 range."""
import os
import numpy as np
from omegaconf import OmegaConf
from emukit.core import ContinuousParameter, ParameterSpace
from src.constants import CONFIG_PATH


def rescale(input: np.ndarray, config_name: str = "sixd") -> np.ndarray:
    """Rescale the numbers to fall in 0.0 to 1.0 range.

    Should write a test that the inverse works.

    Args:
        input (np.ndarray): input array.
        config_name (str, optional): config name. Defaults to "sixd".

    Returns:
        np.ndarray: rescaled array.
    """
    # this will only deal with 1 dimensional arrays at the moment
    config = OmegaConf.load(os.path.join(CONFIG_PATH, config_name + ".yaml"))
    ones = np.ones((input.shape[0]))
    diffs = np.array(
        [config[i].max - config[i].min for i in config]
    )  # .reshape(input.shape[0], 1)
    mins = np.array([config[i].min for i in config])  # .reshape(input.shape[0], 1)
    print(diffs.shape, mins.shape, input.shape)
    return (input - np.dot(ones, mins)) * np.dot(ones, 1 / diffs)


def rescale_inverse(input: np.ndarray, config_name: str = "sixd") -> np.ndarray:
    """Rescale back the numbers to fall in original range.

    Args:
        input (np.ndarray): input array.
        config_name (str, optional): config name. Defaults to "sixd".

    Returns:
        np.ndarray: rescaled array.
    """
    config = OmegaConf.load(os.path.join(CONFIG_PATH, config_name + ".yaml"))
    ones = np.ones((input.shape[0], 1))
    diffs = np.array([config[i].max - config[i].min for i in config])
    mins = np.array([config[i].min for i in config])
    print(diffs.shape, mins.shape, input.shape)
    return np.dot(input, np.dot(ones, diffs)) + np.dot(ones, mins)
