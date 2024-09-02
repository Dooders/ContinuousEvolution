"""
This module contains the implementation of the Agent class and the AgentFactory
class.

The Agent class is a custom PyTorch nn.Module extension representing an agent.
It includes methods for forward pass, state retrieval, and creation from a model.
"""

from typing import Any, Callable, Dict, List, Type, Union

import torch
from torch import nn

from utils import model_hash


class Agent(nn.Module):
    """
    Custom PyTorch nn.Module extension representing an agent.

    Parameters
    ----------
    model : Union[torch.nn.Module, Type[nn.Module]]
        Neural network model instance or model class.
    arguments : Dict[str, Any], optional
        Arguments to pass to the model class if a model class is provided.

    Attributes
    ----------
    model : torch.nn.Module
        Neural network model instance.
    id : str
        Unique identifier for the agent.
    arguments : Dict[str, Any]
        Arguments used to initialize the model.
    fitness : float
        Fitness of the agent.

    Methods
    -------
    forward(x: torch.Tensor) -> torch.Tensor:
        Forward pass through the model.
    state(self) -> Dict[str, Any]:
        Get the state of the agent.
    from_model(cls, model: nn.Module, arguments: Dict[str, Any] = None) -> "Agent":
        Create an agent from a model.
    """

    def __init__(
        self, model: Union[nn.Module, Type[nn.Module]], arguments: Dict[str, Any] = None
    ) -> None:
        super().__init__()
        if isinstance(model, nn.Module):
            # If an instantiated model is provided
            self.model = model
        else:
            # If a model class is provided, instantiate it using the arguments
            if arguments is None:
                raise ValueError(
                    "Arguments must be provided when initializing from a model class."
                )
            self.model = model(**arguments)

        self.arguments = arguments
        self.id = model_hash(self.model)
        self.temporal_id = self.id
        self.fitness = 0
        self.fitness_history = []
        self.output_history = []
        self.input_history = []

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
        output = self.model(x)
        self.output_history.append(output.item())
        self.input_history.append(x.item())
        return output

    def __str__(self) -> str:
        return f"Agent {self.id}"

    def __repr__(self) -> str:
        return self.__str__()

    def state(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "temporal_id": model_hash(self.model),
            "fitness": self.fitness,
            "fitness_history": self.fitness_history,
            "output_history": self.output_history,
            "input_history": self.input_history,
        }

    def evaluate(
        self, fitness_function: Callable[[List[float], List[float]], float]
    ) -> float:
        self.fitness = fitness_function(self.output_history, self.input_history)
        self.fitness_history.append(self.fitness)
        return self.fitness

    @classmethod
    def from_model(cls, model: nn.Module, arguments: Dict[str, Any] = None) -> "Agent":
        return cls(model=model, arguments=arguments)
