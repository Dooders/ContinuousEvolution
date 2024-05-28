import torch
from torch import nn


class Agent(nn.Module):
    """
    Base class for all agents in the simulation.

    Parameters
    ----------
    model : torch.nn.Module
        Neural network model class.
    arguments : dict
        Arguments to pass to the model class.

    Attributes
    ----------
    id : int
        Unique identifier for the agent.

    Methods
    -------
    __call__(x: torch.Tensor) -> torch.Tensor:
        Forward pass through the model.
    """

    def __init__(self, model: torch.nn.Module, arguments: dict) -> None:
        super(Agent, self).__init__()
        self.arguments = arguments
        self.model = model(**arguments)
        self.id = self.model.id

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        return self.model(x)
