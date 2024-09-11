from typing import Tuple

import torch
import torch.nn as nn

from evolution.utils import model_hash


class SimpleSequentialNetwork(nn.Module):
    """
    A simple sequential network with two linear layers and ReLU activation.

    Output layer has a sigmoid activation.

    Methods
    -------
    forward(x)
        Forward pass of the network.
    """

    def __init__(self, input_size: int = 8) -> None:
        super(SimpleSequentialNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 8), nn.ReLU(), nn.Linear(8, 1), nn.Sigmoid()
        )
        self.id = model_hash(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Parameters
        ----------
        x : torch.Tensor
            The input to the network.

        Returns
        -------
        torch.Tensor
            The output of the network.
        """
        return self.net(x)
