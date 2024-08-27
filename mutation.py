"""
Mutation strategies for the continuous evolution algorithm.

The mutation strategy is used to mutate the weights of the neural network.
"""

from abc import ABC, abstractmethod
from typing import Tuple

import torch

from agent import Agent


class MutationStrategy(ABC):
    """
    Abstract base class for mutation strategies.

    Intended to be used as a base class for implementing different mutation
    strategies.

    Methods
    -------
    mutate(agent: "Agent") -> "Agent":
        Perform mutation on the weights of the neural network.
    """

    @classmethod
    @abstractmethod
    def mutate(cls, agent: "Agent") -> "Agent":
        """
        Perform mutation on the weights of the neural network.

        Parameters
        ----------
        agent: "Agent"
            Neural network to be mutated.

        Returns
        -------
        "Agent"
            Neural network with mutated weights.
        """
        raise NotImplementedError("Mutation strategy must implement the mutate method.")


class Uniform(MutationStrategy):
    """
    Each weight of the neural network has a fixed probability of being altered.

    The new value is typically chosen randomly from a uniform distribution over
    a predefined range. This is a straightforward and widely used mutation strategy.

    Methods
    -------
    mutate(agent: "Agent", mutation_rate: float = 0.1, scale: float = 0.05) -> "Agent":
        Perform mutation on the weights of the neural network.
    """

    @classmethod
    def mutate(
        cls, agent: "Agent", mutation_rate: float = 0.1, scale: float = 0.05
    ) -> "Agent":
        """
        Perform mutation on the weights of the neural network.

        Parameters
        ----------
        agent: "Agent"
            Neural network to be mutated.
        mutation_rate: float, optional
            Probability of a weight being mutated, by default 0.1
        scale: float, optional
            Range of the uniform distribution from which the new value is chosen,
            by default 0.05

        Returns
        -------
        "Agent"
            Neural network with mutated weights.
        """
        for param in agent.parameters():
            if torch.rand(1) < mutation_rate:
                noise = torch.rand_like(param) * scale
                param.data += noise


class Gaussian(MutationStrategy):
    """
    Strategy involves adding a small change to the weights, where the change
    follows a Gaussian (or normal) distribution.

    This allows for both small and occasionally larger tweaks in the weights,
    facilitating both fine and coarse tuning of the neural network.

    Methods
    -------
    mutate(agent: "Agent", mutation_rate: float = 0.1, scale: float = 0.05) -> "Agent":
        Perform mutation on the weights of the neural network.
    """

    @classmethod
    def mutate(
        cls, agent: "Agent", mutation_rate: float = 0.1, scale: float = 0.05
    ) -> "Agent":
        """
        Perform mutation on the weights of the neural network.

        Parameters
        ----------
        agent: "Agent"
            Neural network to be mutated.
        mutation_rate: float, optional
            Probability of a weight being mutated, by default 0.1
        scale: float, optional
            Standard deviation of the Gaussian distribution, by default 0.05

        Returns
        -------
        "Agent"
            Neural network with mutated weights.
        """
        for param in agent.parameters():
            if torch.rand(1) < mutation_rate:
                noise = torch.randn_like(param) * scale
                param.data += noise


class NonUniform(MutationStrategy):
    """
    Mutation varies over time, becoming more fine-grained as the number of
    generations increases.

    Initially, it allows for significant changes to the weights to explore a
    broader search space, and later it fine-tunes the solutions by making
    smaller changes.

    Methods
    -------
    mutate(agent: "Agent", mutation_rate: float = 0.1, scale: float = 0.05, max_generations: int = 1000) -> "Agent":
        Perform mutation on the weights of the neural network.

    #! Not working
    """

    @classmethod
    def mutate(
        cls,
        agent: "Agent",
        mutation_rate: float = 0.1,
        scale: float = 0.05,
        max_generations: int = 1000,
    ) -> "Agent":
        """
        Perform mutation on the weights of the neural network.

        Parameters
        ----------
        agent: "Agent"
            Neural network to be mutated.
        mutation_rate: float, optional
            Probability of a weight being mutated, by default 0.1
        scale: float, optional
            Range of the uniform distribution from which the new value is chosen,
            by default 0.05
        max_generations: int, optional
            Maximum number of generations, by default 1000

        Returns
        -------
        "Agent"
            Neural network with mutated weights.
        """
        generation = 0
        for param in agent.parameters():
            if torch.rand(1) < mutation_rate * (1 - generation / max_generations):
                noise = torch.randn_like(param) * scale
                param.data += noise

        generation += 1


class Polynomial(MutationStrategy):
    """
    Provides a way to control the distribution of mutations, allowing the
    algorithm to fine-tune solutions more effectively.

    It uses a polynomial probability distribution to decide the magnitude of mutation.

    Methods
    -------
    mutate(agent: "Agent", mutation_rate: float = 0.1, scale: float = 0.05) -> "Agent":
        Perform mutation on the weights of the neural network.
    """

    @classmethod
    def mutate(
        cls, agent: "Agent", mutation_rate: float = 0.1, scale: float = 0.05
    ) -> "Agent":
        """
        Perform mutation on the weights of the neural network.

        Parameters
        ----------
        agent: "Agent"
            Neural network to be mutated.
        mutation_rate: float, optional
            Probability of a weight being mutated, by default 0.1
        scale: float, optional
            Range of the uniform distribution from which the new value is chosen,
            by default 0.05

        Returns
        -------
        "Agent"
            Neural network with mutated weights.
        """
        for param in agent.parameters():
            if torch.rand(1) < mutation_rate:
                noise = torch.randn_like(param) * scale
                delta = noise * (2 * torch.rand_like(param) - 1)
                param.data += delta


class BitFlip(MutationStrategy):
    """
    Solutions are encoded in binary, bit flip mutation can be adapted for neural
    networks by considering a binary representation of the weights.

    Each bit has a probability of being flipped (changed from 0 to 1, or vice versa).

    Methods
    -------
    mutate(agent: "Agent", mutation_rate: float = 0.1) -> "Agent":
        Perform mutation on the weights of the neural network.
    """

    @classmethod
    def mutate(cls, agent: "Agent", mutation_rate: float = 0.1) -> "Agent":
        """
        Perform mutation on the weights of the neural network.

        Parameters
        ----------
        agent: "Agent"
            Neural network to be mutated.
        mutation_rate: float, optional
            Probability of a weight being mutated, by default 0.1

        Returns
        -------
        "Agent"
            Neural network with mutated weights.
        """
        for param in agent.parameters():
            if torch.rand(1) < mutation_rate:
                mask = torch.randint(0, 2, param.size(), device=param.device)
                param.data = param.data ^ mask


class Boundary(MutationStrategy):
    """
    Strategy specifically targets the boundaries of the parameter space.

    If a mutation is to occur, it either sets the parameter to its upper or lower
    boundary value.

    This can be particularly useful when the optimal parameters are suspected to
    lie near the edges of the parameter range.

    Methods
    -------
    mutate(agent: "Agent", mutation_rate: float = 0.1, upper_bound: float = 1.0, lower_bound: float = -1.0) -> "Agent":
        Perform mutation on the weights of the neural network.
    """

    @classmethod
    def mutate(
        cls,
        agent: "Agent",
        mutation_rate: float = 0.1,
        upper_bound: float = 1.0,
        lower_bound: float = -1.0,
    ) -> "Agent":
        """
        Perform mutation on the weights of the neural network.

        Parameters
        ----------
        agent: "Agent"
            Neural network to be mutated.
        mutation_rate: float, optional
            Probability of a weight being mutated, by default 0.1
        upper_bound: float, optional
            Upper boundary value, by default 1.0
        lower_bound: float, optional
            Lower boundary value, by default -1.0

        Returns
        -------
        "Agent"
            Neural network with mutated weights.
        """
        for param in agent.parameters():
            if torch.rand(1) < mutation_rate:
                mask = torch.rand_like(param) < 0.5
                param.data = torch.where(
                    mask,
                    torch.tensor(lower_bound, device=param.device),
                    torch.tensor(upper_bound, device=param.device),
                )


class AdaptiveGaussian(MutationStrategy):
    """
    Adjust the mutation parameters dynamically based on the performance of the
    population across generations.

    The idea is to increase mutation rates when the population appears to be
    converging prematurely (to escape local minima) and to decrease them as the
    solutions approach an optimum to maintain stability.

    Methods
    -------
    update_rate(current_performance) -> None:
        Update the mutation rate based on the current performance.
    mutate(agent: "Agent", mutation_rate: float = 0.1, scale: float = 0.05) -> "Agent":
        Perform mutation on the weights of the neural network.
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

    @classmethod
    def mutate(
        cls, agent: "Agent", mutation_rate: float = 0.1, scale: float = 0.05
    ) -> "Agent":
        """
        #! Needs to be finished
        Perform mutation on the weights of the neural network.

        Parameters
        ----------
        agent: "Agent"
            Neural network to be mutated.
        mutation_rate: float, optional
            Probability of a weight being mutated, by default 0.1
        scale: float, optional
            Standard deviation of the Gaussian distribution, by default 0.05

        Returns
        -------
        "Agent"
            Neural network with mutated weights.
        """

        for param in agent.parameters():
            if torch.rand(1) < mutation_rate:
                noise = torch.randn_like(param) * scale
                param.data += noise


class Hybrid(MutationStrategy):
    """
    Combine different mutation mechanisms to take advantage of the benefits of each.

    For example, you might use both Gaussian and uniform mutations, where Gaussian
    provides small, fine-tuned changes, and uniform allows for occasional large jumps.

    Methods
    -------
    mutate(agent: "Agent", rate_gaussian: float = 0.1, scale_gaussian: float = 0.05, rate_uniform: float = 0.1, range_uniform: Tuple[float, float] = (-0.1, 0.1)) -> "Agent":
        Perform mutation on the weights of the neural network.
    """

    @classmethod
    def mutate(
        cls,
        agent: "Agent",
        rate_gaussian: float = 0.1,
        scale_gaussian: float = 0.05,
        rate_uniform: float = 0.1,
        range_uniform: Tuple[float, float] = (-0.1, 0.1),
    ) -> "Agent":
        """
        Perform mutation on the weights of the neural network.

        Parameters
        ----------
        agent: "Agent"
            Neural network to be mutated.
        rate_gaussian: float, optional
            Probability of a weight being mutated by Gaussian noise, by default 0.1
        scale_gaussian: float, optional
            Standard deviation of the Gaussian distribution, by default 0.05
        rate_uniform: float, optional
            Probability of a weight being mutated by uniform noise, by default 0.1
        range_uniform: Tuple[float, float], optional
            Range of the uniform distribution from which the new value is chosen, by default (-0.1, 0.1)

        Returns
        -------
        "Agent"
            Neural network with mutated weights.
        """
        for param in agent.parameters():
            if torch.rand(1) < rate_gaussian:
                param.data += scale_gaussian * torch.randn_like(param)
            if torch.rand(1) < rate_uniform:
                param.data += torch.empty_like(param).uniform_(*range_uniform)
