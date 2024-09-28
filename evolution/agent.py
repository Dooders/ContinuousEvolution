"""
The Agent class is a custom PyTorch nn.Module extension representing an agent.
It includes methods for forward pass, state retrieval, and creation from a model.
"""

from typing import Any, Callable, Dict, List, Type, Union

import numpy as np
import torch
from torch import nn
import math

from evolution.utils import experiment_logger as logger
from evolution.utils import model_hash


class ScalingLayer(nn.Module):
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return x * self.scale_factor


class LearnableScalingLayer(nn.Module):
    def __init__(self, initial_scale=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(initial_scale))

    def forward(self, x):
        return x * self.scale


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
        self, model: Union[nn.Module, Type[nn.Module]], arguments: Dict[str, Any] = None,
        scaling_type: str = 'none', scaling_factor: float = 1.0
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

        # Add scaling layer
        if scaling_type == 'fixed':
            self.scaling_layer = ScalingLayer(scaling_factor)
        elif scaling_type == 'learnable':
            self.scaling_layer = LearnableScalingLayer(scaling_factor)
        else:
            self.scaling_layer = nn.Identity()

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        """
        Initialize the weights of the model using He initialization,
        but with a larger scale factor to potentially produce larger outputs.
        """
        for module in self.model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # He initialization with a larger scale factor
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                module.weight.data *= 2.0  # Increase the scale of weights
                
                if module.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(module.bias, -bound, bound)
                    module.bias.data *= 2.0  # Increase the scale of biases

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
        scaled_output = self.scaling_layer(output)
        self.output_history.append(scaled_output.item())
        self.input_history.append(x.tolist())
        logger.info(f"Agent {self.id} output: {scaled_output}")
        return scaled_output

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
