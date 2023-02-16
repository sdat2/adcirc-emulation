"""Rescale the parameter inputs to fall in 0.0 to 1.0 range.

This is used to rescale the inputs to the GP, and then rescale the outputs back to the original range.

The parameters are defined in the config file, and the rescaling is done by subtracting the minimum value, and dividing by the range.

Which config file can be specified, but the default is the sixd.yaml config file.

# TODO: fix to allow some sort of parralization, and ability to identify which axes is which?

TODO: add a test for inverse.
"""
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

    #TODO: add a test that all outputs are between 0 and 1, otherwise raise an error

    """
    # this will only deal with 1 dimensional arrays at the moment
    print("input", input)
    config = OmegaConf.load(os.path.join(CONFIG_PATH, config_name + ".yaml"))
    ones = np.ones((input.shape[0]))
    diffs = np.array(
        [config[i].max - config[i].min for i in config]
    )  # .reshape(input.shape[0], 1)
    print("diffs", diffs)
    mins = np.array([config[i].min for i in config])  # .reshape(input.shape[0], 1)
    print("mins", mins)
    print(diffs.shape, mins.shape, input.shape, ones.shape)
    # return (input - np.dot(ones, mins)) * np.dot(ones, 1 / diffs)
    output = []
    for i in range(input.shape[0]):
        print(input[i], mins[i], diffs[i])
        output.append((input[i] - mins[i]) / diffs[i])
    print("Output", output)
    return (input - mins) / diffs


def rescale_inverse(input: np.ndarray, config_name: str = "sixd") -> np.ndarray:
    """Rescale back the numbers to fall in original range.

    Args:
        input (np.ndarray): input array.
        config_name (str, optional): config name. Defaults to "sixd".

    Returns:
        np.ndarray: rescaled array.
    """
    config = OmegaConf.load(os.path.join(CONFIG_PATH, config_name + ".yaml"))
    ones = np.ones((input.shape[0]))  # , 1
    diffs = np.array([config[i].max - config[i].min for i in config])
    mins = np.array([config[i].min for i in config])
    print(diffs.shape, mins.shape, input.shape, ones.shape)
    return input * diffs + mins
    # return np.dot(input, np.dot(ones, diffs)) + np.dot(ones, mins)
