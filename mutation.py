from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn as nn


class MutationStrategy(ABC):
    @abstractmethod
    def mutate(self, network: nn.Module):
        pass


class UniformMutation(MutationStrategy):
    """
    Each weight of the neural network has a fixed probability of being altered.

    The new value is typically chosen randomly from a uniform distribution over
    a predefined range. This is a straightforward and widely used mutation strategy.
    """

    def __init__(self, mutation_rate: float = 0.1, scale: float = 0.05) -> None:
        self.mutation_rate = mutation_rate
        self.scale = scale

    def mutate(self, network: nn.Module) -> None:
        for param in network.parameters():
            if torch.rand(1) < self.mutation_rate:
                noise = torch.rand_like(param) * self.scale
                param.data += noise


class GaussianMutation(MutationStrategy):
    """
    Strategy involves adding a small change to the weights, where the change
    follows a Gaussian (or normal) distribution.

    This allows for both small and occasionally larger tweaks in the weights,
    facilitating both fine and coarse tuning of the neural network.
    """

    def __init__(self, mutation_rate: float = 0.1, scale: float = 0.05) -> None:
        self.mutation_rate = mutation_rate
        self.scale = scale

    def mutate(self, network: nn.Module) -> None:
        for param in network.parameters():
            if torch.rand(1) < self.mutation_rate:
                noise = torch.randn_like(param) * self.scale
                param.data += noise


class NonUniformMutation(MutationStrategy):
    """
    Mutation varies over time, becoming more fine-grained as the number of
    generations increases.

    Initially, it allows for significant changes to the weights to explore a
    broader search space, and later it fine-tunes the solutions by making
    smaller changes.

    #! Not working
    """

    def __init__(
        self,
        mutation_rate: float = 0.1,
        scale: float = 0.05,
        max_generations: int = 1000,
    ) -> None:
        self.mutation_rate = mutation_rate
        self.scale = scale
        self.max_generations = max_generations
        self.generation = 0

    def mutate(self, network: nn.Module) -> None:
        for param in network.parameters():
            if torch.rand(1) < self.mutation_rate * (
                1 - self.generation / self.max_generations
            ):
                noise = torch.randn_like(param) * self.scale
                param.data += noise

        self.generation += 1


class PolynomialMutation(MutationStrategy):
    """
    Provides a way to control the distribution of mutations, allowing the
    algorithm to fine-tune solutions more effectively.

    It uses a polynomial probability distribution to decide the magnitude of mutation.
    """

    def __init__(
        self, mutation_rate: float = 0.1, scale: float = 0.05, eta: float = 20
    ) -> None:
        self.mutation_rate = mutation_rate
        self.scale = scale
        self.eta = eta

    def mutate(self, network: nn.Module) -> None:
        for param in network.parameters():
            if torch.rand(1) < self.mutation_rate:
                noise = torch.randn_like(param) * self.scale
                delta = noise * (2 * torch.rand_like(param) - 1)
                param.data += delta


class BitFlipMutation(MutationStrategy):
    """
    Solutions are encoded in binary, bit flip mutation can be adapted for neural
    networks by considering a binary representation of the weights.

    Each bit has a probability of being flipped (changed from 0 to 1, or vice versa).
    """

    def __init__(self, mutation_rate: float = 0.1) -> None:
        self.mutation_rate = mutation_rate

    def mutate(self, network: nn.Module) -> None:
        for param in network.parameters():
            if torch.rand(1) < self.mutation_rate:
                mask = torch.randint(0, 2, param.size(), device=param.device)
                param.data = param.data ^ mask


class BoundaryMutation(MutationStrategy):
    """
    Strategy specifically targets the boundaries of the parameter space.

    If a mutation is to occur, it either sets the parameter to its upper or lower
    boundary value.

    This can be particularly useful when the optimal parameters are suspected to
    lie near the edges of the parameter range.
    """

    def __init__(
        self,
        mutation_rate: float = 0.1,
        lower_bound: float = -1.0,
        upper_bound: float = 1.0,
    ) -> None:
        self.mutation_rate = mutation_rate
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def mutate(self, network: nn.Module) -> None:
        for param in network.parameters():
            if torch.rand(1) < self.mutation_rate:
                mask = torch.rand_like(param) < 0.5
                param.data = torch.where(
                    mask,
                    torch.tensor(self.lower_bound, device=param.device),
                    torch.tensor(self.upper_bound, device=param.device),
                )


class AdaptiveGaussianMutation(MutationStrategy):
    """
    Adjust the mutation parameters dynamically based on the performance of the
    population across generations.

    The idea is to increase mutation rates when the population appears to be
    converging prematurely (to escape local minima) and to decrease them as the
    solutions approach an optimum to maintain stability.
    """

    def __init__(
        self,
        initial_rate: float = 0.1,
        min_rate: float = 0.01,
        max_rate: float = 0.5,
        scale: float = 0.05,
    ) -> None:
        self.rate = initial_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.scale = scale
        self.performance_tracker = []

    def update_rate(self, current_performance) -> None:
        if current_performance > min(self.performance_tracker):
            self.rate = max(self.min_rate, self.rate * 0.9)
        else:
            self.rate = min(self.max_rate, self.rate * 1.1)
        self.performance_tracker.append(current_performance)

    def mutate(self, network: nn.Module) -> None:
        for param in network.parameters():
            if torch.rand(1) < self.rate:
                noise = torch.randn_like(param) * self.scale
                param.data += noise


class HybridMutation(MutationStrategy):
    """
    Combine different mutation mechanisms to take advantage of the benefits of each.

    For example, you might use both Gaussian and uniform mutations, where Gaussian
    provides small, fine-tuned changes, and uniform allows for occasional large jumps.
    """

    def __init__(
        self,
        rate_gaussian: float = 0.1,
        scale_gaussian: float = 0.05,
        rate_uniform: float = 0.1,
        range_uniform: Tuple[float, float] = (-0.1, 0.1),
    ) -> None:
        self.rate_gaussian = rate_gaussian
        self.scale_gaussian = scale_gaussian
        self.rate_uniform = rate_uniform
        self.range_uniform = range_uniform

    def mutate(self, network: nn.Module) -> None:
        for param in network.parameters():
            if torch.rand(1) < self.rate_gaussian:
                param.data += self.scale_gaussian * torch.randn_like(param)
            if torch.rand(1) < self.rate_uniform:
                param.data += torch.empty_like(param).uniform_(*self.range_uniform)
