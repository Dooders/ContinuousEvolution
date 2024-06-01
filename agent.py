from typing import Any, Dict, Type

import torch
from torch import nn

from utils import model_hash


class Agent(nn.Module):
    """
    Custom pytorch nn.Module extension representing an agent.

    Parameters
    ----------
    model : torch.nn.Module
        Neural network model class.
    arguments : Dict[str, Any]
        Arguments to pass to the model class.

    Attributes
    ----------
    model : torch.nn.Module
        Neural network model instance.
    id : str
        Unique identifier for the agent.

    Methods
    -------
    forward(x: torch.Tensor) -> torch.Tensor:
        Forward pass through the model.
    """

    def __init__(self, model_cls: Type[nn.Module], arguments: Dict[str, Any]) -> None:
        super().__init__()
        try:
            self.arguments = arguments
            self.model = model_cls(**arguments)
            self.id = model_hash(self.model)
            self.fitness = 0
        except TypeError as e:
            raise ValueError(
                f"Error initializing model with arguments {arguments}: {e}"
            )
        except Exception as e:
            raise ValueError(f"Unexpected error initializing model: {e}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        #! Add state update logic to update age, etc.
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

    def __str__(self) -> str:
        return f"Agent {self.id}"

    def __repr__(self) -> str:
        return self.__str__()


class AgentFactory:
    """
    Factory class for creating agents.

    Methods
    -------
    create(arguments: Dict[str, Any]) -> Agent:
        Create an agent with the given arguments.
    """

    @classmethod
    def create(self, model_cls: Type[nn.Module], arguments: Dict[str, Any]) -> "Agent":
        """
        Create an agent with the given arguments.

        Parameters
        ----------
        model_cls : torch.nn.Module
            Neural network model class. i.e. SimpleSequentialNetwork
        arguments : Dict[str, Any]
            Arguments to pass to the model class.

        Returns
        -------
        Agent
            Agent instance based on provided arguments.
        """
        try:
            return Agent(model_cls, arguments)
        except ValueError as e:
            print(f"Error creating agent: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error creating agent: {e}")
            raise
