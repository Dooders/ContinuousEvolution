import hashlib

import numpy as np
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
