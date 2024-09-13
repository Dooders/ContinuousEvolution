import hashlib
import logging
import os
import time
from typing import Union

import numpy as np
import torch
import torch.nn as nn


def model_hash(model: nn.Module) -> str:
    """
    Generate a unique hash for a PyTorch model.

    Parameters
    ----------
    model : nn.Module
        The model to hash.

    Returns
    -------
    str
        The hash of the model parameters.
    """
    param_string = "".join(
        [str(p.data.cpu().numpy().byteswap().tobytes()) for p in model.parameters()]
    )
    return hashlib.md5(param_string.encode() + str(time.time()).encode()).hexdigest()


def extract_parameters(model: nn.Module) -> np.ndarray:
    """
    Extract the parameters from a model. (It's weights and biases)

    Returns a flattened list of parameters. First the weights, then the biases.
    Then flattens that into a single list.

    Parameters
    ----------
    model : nn.Module
        The model from which to extract the parameters.

    Returns
    -------
    np.ndarray
        The extracted parameters in a single list.
    """
    parameters = []
    for layer in model.net:
        if isinstance(layer, nn.Linear):
            parameters.append(layer.weight.data.numpy().flatten())
            parameters.append(layer.bias.data.numpy().flatten())
    return np.concatenate(parameters)


def set_seed(seed: Union[str, int]) -> None:
    """
    Set the seed for the random number generators.

    Parameters
    ----------
    seed : Union[str, int]
        The seed to set. If a string is provided, it will be converted to a 32-bit integer hash.
    """
    if isinstance(seed, str):
        # Convert string to a 32-bit integer hash
        seed = int(hashlib.md5(seed.encode()).hexdigest(), 16) % (2**32)
    else:
        seed = int(seed) % (2**32)
    torch.manual_seed(seed)
    np.random.seed(seed)


def _experiment_logger() -> logging.Logger:
    """
    Set up a logger that only writes to a file, not to the console.
    Deletes the existing log file if it exists.

    Returns
    -------
    logging.Logger
        The configured logger.
    """
    log_file = "simulation.log"

    # Delete the existing log file if it exists
    if os.path.exists(log_file):
        os.remove(log_file)

    # Set up logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(message)s")

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Prevent the logger from propagating messages to the root logger
    logger.propagate = False

    return logger


experiment_logger = _experiment_logger()
