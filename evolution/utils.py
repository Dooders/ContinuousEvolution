import hashlib
from typing import Union

import numpy as np
import torch
import torch.nn as nn


def model_hash(model: nn.Module) -> str:
    """
    Generate a hash of the model parameters.

    Parameters
    ----------
    model : nn.Module
        The model to hash.

    Returns
    -------
    str
        The hash of the model parameters.
    """
    hash_md5 = hashlib.md5()
    extracted_parameters = extract_parameters(model)
    hash_md5.update(extracted_parameters.tobytes())
    return hash_md5.hexdigest()


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
