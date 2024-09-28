"""
The Agent class is a custom PyTorch nn.Module extension representing an agent.
It includes methods for forward pass, state retrieval, and creation from a model.
"""

from typing import Any, Callable, Dict, List, Type, Union

import numpy as np
import torch
from torch import nn

from evolution.utils import experiment_logger as logger
from evolution.utils import model_hash


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
    evaluate(self, fitness_function: Callable[[List[float], List[float]], float], target_history: List[float]) -> float:
        Calculate the fitness of the agent.
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
        self.input_history.append(x.tolist())
        logger.info(f"Agent {self.id} output: {output}")
        return output

    def __str__(self) -> str:
        return f"Agent {self.id}"

    def __repr__(self) -> str:
        return self.__str__()

    def state(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "output": self.output_history[-1] if self.output_history else None,
            # ... other state properties ...
        }

    def evaluate(
        self,
        fitness_function: Callable[[List[float], List[float]], float],
        X: torch.Tensor,
        # y: torch.Tensor,
    ) -> float:
        """
        Calculate the fitness of the agent.

        Parameters
        ----------
        fitness_function : Callable[[List[float], List[float]], float]
            Function to calculate the fitness of the agent.
        X : torch.Tensor
            Input tensor.
        y : torch.Tensor
            Target tensor.

        Returns
        -------
        float
            Fitness of the agent.
        """
        if self.output_history:
            score = fitness_function(self, X)
        else:
            score = 0
        self.fitness_history.append(score)
        logger.info(
            f"Evaluating agent {self.id} fitness: {score}, fitness history: {self.fitness_history}"
        )
        return score

    @property
    def fitness(self) -> float:
        """
        Calculate the fitness of the agent.

        Returns
        -------
        float
            Fitness of the agent. 0 if no fitness history.
        """
        if not self.fitness_history:
            return 0
        return np.mean(self.fitness_history)

    @property
    def output(self) -> float:
        """
        Return the last output of the agent.

        Returns
        -------
        float
            Last output of the agent. None if no output history.
        """
        if not self.output_history:
            return None
        return self.output_history[-1]

    @classmethod
    def from_model(cls, model: nn.Module, arguments: Dict[str, Any] = None) -> "Agent":
        return cls(model=model, arguments=arguments)

    def update_id(self):
        """Update the agent's ID based on its current state."""
        self.id = model_hash(self.model)
        self.temporal_id = self.id
