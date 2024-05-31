from abc import ABC, abstractmethod

import torch

from agent import Agent


class CrossoverStrategy(ABC):
    """
    Abstract base class for crossover strategies.

    Intended to be used as a base class for implementing different crossover
    strategies.

    Methods
    -------
    crossover(parent_x: "Agent", parent_y: "Agent", settings: dict) -> "Agent":
        Perform crossover between two parent networks to create a child network.
    """

    @abstractmethod
    def crossover(
        self, parent_x: "Agent", parent_y: "Agent", settings: dict = None
    ) -> "Agent":
        """
        Perform crossover between two parent networks to create a child network.

        Parameters
        ----------
        parent_x : "Agent"
            First parent network.
        parent_y : "Agent"
            Second parent network.
        settings : dict
            Settings for the child network. None by default.

        Returns
        -------
        "Agent"
            Child network created by crossover.
        """

        raise NotImplementedError(
            "Crossover strategy must implement the crossover method."
        )


class Average(CrossoverStrategy):
    """
    Crossover strategy that averages the weights of two parent networks.
    """

    def crossover(
        self, parent_x: "Agent", parent_y: "Agent", settings: dict = None
    ) -> "Agent":
        """
        Perform crossover between two parent networks to create a child network.

        Parameters
        ----------
        parent_x : "Agent"
            First parent network.
        parent_y : "Agent"
            Second parent network.
        settings : dict
            Settings for the child network. None by default.

        Returns
        -------
        "Agent"
            Child network created by crossover.
        """
        settings = parent_x.arguments if settings is None else settings
        child = type(parent_x.model)(**settings)
        for child_param, param_x, param_y in zip(
            child.parameters(), parent_x.parameters(), parent_y.parameters()
        ):
            child_param.data.copy_((param_x.data + param_y.data) / 2.0)
        return child


class Random(CrossoverStrategy):
    """
    Crossover strategy that randomly selects weights from two parent networks.
    """

    def crossover(
        self, parent_x: "Agent", parent_y: "Agent", settings: dict = None
    ) -> "Agent":
        """
        Perform crossover between two parent networks to create a child network.

        Parameters
        ----------
        parent_x : "Agent"
            First parent network.
        parent_y : "Agent"
            Second parent network.
        settings : dict
            Settings for the child network. None by default.

        Returns
        -------
        "Agent"
            Child network created by crossover.
        """
        settings = parent_x.arguments if settings is None else settings
        child = type(parent_x.model)(**settings)
        for child_param, param__x, param__y in zip(
            child.parameters(), parent_x.parameters(), parent_y.parameters()
        ):
            mask = torch.rand(param__x.size()) > 0.5
            child_param.data.copy_(torch.where(mask, param__x.data, param__y.data))
        return child


class RandomPoint(CrossoverStrategy):
    """
    Crossover strategy that randomly selects a point and swaps weights from two
    parent networks.
    """

    def crossover(
        self, parent_x: "Agent", parent_y: "Agent", settings: dict = None
    ) -> "Agent":
        """
        Perform crossover between two parent networks to create a child network.

        Parameters
        ----------
        parent_x : "Agent"
            First parent network.
        parent_y : "Agent"
            Second parent network.
        settings : dict
            Settings for the child network. None by default.

        Returns
        -------
        "Agent"
            Child network created by crossover.
        """
        settings = parent_x.arguments if settings is None else settings
        child = type(parent_x.model)(**settings)
        for child_param, param__x, param_y in zip(
            child.parameters(), parent_x.parameters(), parent_y.parameters()
        ):
            if (
                param__x.dim() > 1
            ):  # Check if the parameter tensor has more than one dimension
                mask = torch.zeros(
                    param__x.size(), dtype=torch.bool
                )  # Initialize mask as a boolean tensor
                split_point = torch.randint(
                    0, param__x.size(1), (1,)
                ).item()  # Randomly select a split point
                mask[:, :split_point] = True  # Set the first part of the mask to True
                child_param.data.copy_(torch.where(mask, param__x.data, param_y.data))
            else:
                # For 1-dimensional tensors, copy the entire tensor from one of the parents randomly
                if torch.rand(1) < 0.5:
                    child_param.data.copy_(param__x.data)
                else:
                    child_param.data.copy_(param_y.data)
        return child


class RandomRange(CrossoverStrategy):
    """
    Crossover strategy that randomly selects a range and swaps weights from two
    parent networks.
    """

    def crossover(
        self, parent_x: "Agent", parent_y: "Agent", settings: dict = None
    ) -> "Agent":
        """
        Perform crossover between two parent networks to create a child network.

        Parameters
        ----------
        parent_x : "Agent"
            First parent network.
        parent_y : "Agent"
            Second parent network.
        settings : dict
            Settings for the child network. None by default.

        Returns
        -------
        "Agent"
            Child network created by crossover.
        """
        settings = parent_x.arguments if settings is None else settings
        child = type(parent_x.model)(**settings)
        for child_param, param_x, param_y in zip(
            child.parameters(), parent_x.parameters(), parent_y.parameters()
        ):
            if param_x.dim() > 1:
                mask = torch.zeros(param_x.size(), dtype=torch.bool)
                start = torch.randint(0, param_x.size(1), (1,)).item()
                end = torch.randint(start, param_x.size(1), (1,)).item()
                mask[:, start:end] = True
                child_param.data.copy_(torch.where(mask, param_x.data, param_y.data))
            else:
                if torch.rand(1) < 0.5:
                    child_param.data.copy_(param_x.data)
                else:
                    child_param.data.copy_(param_y.data)
        return child
