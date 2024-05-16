from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class CrossoverStrategy(ABC):
    @abstractmethod
    def crossover(self, parent1: nn.Module, parent2: nn.Module) -> nn.Module:
        raise NotImplementedError(
            "Crossover strategy must implement the crossover method."
        )


class AverageCrossover(CrossoverStrategy):
    """
    Crossover strategy that averages the weights of two parent networks.
    """

    def crossover(self, parent1: nn.Module, parent2: nn.Module) -> nn.Module:
        child = type(parent1)()
        for child_param, param1, param2 in zip(
            child.parameters(), parent1.parameters(), parent2.parameters()
        ):
            child_param.data.copy_((param1.data + param2.data) / 2.0)
        return child


class RandomCrossover(CrossoverStrategy):
    """
    Crossover strategy that randomly selects weights from two parent networks.
    """

    def crossover(self, parent1: nn.Module, parent2: nn.Module) -> nn.Module:
        child = type(parent1)()
        for child_param, param1, param2 in zip(
            child.parameters(), parent1.parameters(), parent2.parameters()
        ):
            mask = torch.rand(param1.size()) > 0.5
            child_param.data.copy_(torch.where(mask, param1.data, param2.data))
        return child


class RandomPointCrossover(CrossoverStrategy):
    """
    Crossover strategy that randomly selects a point and swaps weights from two
    parent networks.
    """

    def crossover(self, parent1: nn.Module, parent2: nn.Module) -> nn.Module:
        child = type(parent1)()
        for child_param, param1, param2 in zip(
            child.parameters(), parent1.parameters(), parent2.parameters()
        ):
            if (
                param1.dim() > 1
            ):  # Check if the parameter tensor has more than one dimension
                mask = torch.zeros(
                    param1.size(), dtype=torch.bool
                )  # Initialize mask as a boolean tensor
                split_point = torch.randint(
                    0, param1.size(1), (1,)
                ).item()  # Randomly select a split point
                mask[:, :split_point] = True  # Set the first part of the mask to True
                child_param.data.copy_(torch.where(mask, param1.data, param2.data))
            else:
                # For 1-dimensional tensors, copy the entire tensor from one of the parents randomly
                if torch.rand(1) < 0.5:
                    child_param.data.copy_(param1.data)
                else:
                    child_param.data.copy_(param2.data)
        return child


class RandomRangeCrossover(CrossoverStrategy):
    """
    Crossover strategy that randomly selects a range and swaps weights from two
    parent networks.
    """

    def crossover(self, parent1: nn.Module, parent2: nn.Module) -> nn.Module:
        child = type(parent1)()
        for child_param, param1, param2 in zip(
            child.parameters(), parent1.parameters(), parent2.parameters()
        ):
            if param1.dim() > 1:
                mask = torch.zeros(param1.size(), dtype=torch.bool)
                start = torch.randint(0, param1.size(1), (1,)).item()
                end = torch.randint(start, param1.size(1), (1,)).item()
                mask[:, start:end] = True
                child_param.data.copy_(torch.where(mask, param1.data, param2.data))
            else:
                if torch.rand(1) < 0.5:
                    child_param.data.copy_(param1.data)
                else:
                    child_param.data.copy_(param2.data)
        return child
