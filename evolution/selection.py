"""
This module contains the implementation of the Selection class and its subclasses.

The Selection class is an abstract class that defines the interface for selection methods.

The subclasses implement different selection methods such as roulette wheel, tournament,
rank, elitism, stochastic universal sampling, truncation, Boltzmann, softmax, and fitness scaling.
"""

import random
from abc import ABC, abstractmethod
from typing import List, Tuple

from evolution.agent import Agent
from evolution.population import Population


class Selection(ABC):
    """
    Abstract class for selection methods.

    Intended to be used as a base class to be inherited where the select method
    returns a Population object of the selected agents.

    A Selection subclass is passed to the ContinuousEvolution object during initialization.

    Parameters
    ----------
    population : Population[Agent]
        The population of agents to select from.
    n : int
        The number of agents to select.

    Returns
    -------
    Population[Agent]
        The selected population of agents.
    """

    @classmethod
    @abstractmethod
    def select(self, population: "Population[Agent]", n: int) -> "Population[Agent]":
        """
        Select n agents from the population.

        Parameters
        ----------
        population : "Population[Agent]"
            List of agents in the population, sorted desc by fitness.
        n : int
            Number of agents to select.

        Returns
        -------
        "Population[Agent]"
            List of selected agents of length n.
        """
        raise NotImplementedError(
            "Select method not implemented. See documentation for details."
        )


class RouletteWheel(Selection):
    """
    Selecting individuals based on their fitness relative to the population.

    The probability of an individual being selected is proportional to its fitness.

    This simulates a roulette wheel where each individual occupies a slice of the
    wheel proportional to its fitness score.
    """

    @classmethod
    def select(cls, population: "Population[Agent]", n: int) -> "Population[Agent]":
        """
        Select n agents from the population.

        Parameters
        ----------
        population : "Population[Agent]"
            List of agents in the population, sorted desc by fitness.
        n : int
            Number of agents to select.

        Returns
        -------
        "Population[Agent]"
            List of selected agents of length n.
        """
        total_fitness = sum(population.fitness)
        # Avoid division by zero; randomly choose an agent
        if total_fitness == 0:
            return [random.choice(population) for _ in range(n)]
        probabilities = [f / total_fitness for f in population.fitnesses]
        selected = random.choices(population, probabilities, k=n)
        return selected


class Tournament(Selection):
    """
    Selecting individuals based on a tournament selection process.

    The tournament selection process involves selecting k individuals from the population
    and then selecting the best individual from the k individuals.
    """

    @classmethod
    def select(
        cls, population: "Population[Agent]", n: int, k: int
    ) -> "Population[Agent]":
        """
        Select n agents from the population.

        Parameters
        ----------
        population : "Population[Agent]"
            List of agents in the population, sorted desc by fitness.
        n : int
            Number of agents to select.
        k : int
            Number of agents to select in each tournament.

        Returns
        -------
        "Population[Agent]"
            List of selected agents of length n.
        """
        selected = []
        for _ in range(n):
            tournament = random.sample(list(zip(population, population.fitness)), k)
            winner = max(tournament, key=lambda x: x[1])[0]
            selected.append(winner)
        return selected


class Rank(Selection):
    """
    Selecting individuals based on their rank in the population.

    The rank selection process involves sorting the population based on fitness
    and then selecting individuals based on their rank.
    """

    @classmethod
    def select(cls, population: List[Tuple[Agent, float]], n: int) -> List[Agent]:
        """
        Select n agents from the population.

        Parameters
        ----------
        population : List[Tuple[Agent, float]]
            List of tuples containing (agent, fitness_score)
        n : int
            Number of agents to select.

        Returns
        -------
        List[Agent]
            List of selected agents of length n.
        """
        ranked_population = sorted(population, key=lambda x: x[1], reverse=True)
        selected = [agent for agent, _ in ranked_population[:n]]
        return selected


class Elitism(Selection):
    """
    #! How is this different from Rank
    Selecting individuals based on elitism.

    The elitism selection process involves selecting the best individuals from the population.
    """

    @classmethod
    def select(cls, population: "Population[Agent]", n: int) -> "Population[Agent]":
        """
        Select n agents from the population.

        Parameters
        ----------
        population : "Population[Agent]"
            List of agents in the population, sorted desc by fitness.
        n : int
            Number of agents to select.

        Returns
        -------
        "Population[Agent]"
            List of selected agents of length n.
        """
        ranked_population = sorted(
            list(zip(population, population.fitness)), key=lambda x: x[1], reverse=True
        )
        selected = [agent for agent, _ in ranked_population[:n]]
        return selected


class StochasticUniversalSampling(Selection):
    """
    Selecting individuals based on stochastic universal sampling.

    The stochastic universal sampling selection process involves selecting individuals
    based on their fitness and a random starting point.
    """

    @classmethod
    def select(cls, population: "Population[Agent]", n: int) -> "Population[Agent]":
        """
        Select n agents from the population.

        Parameters
        ----------
        population : "Population[Agent]"
            List of agents in the population, sorted desc by fitness.
        n : int
            Number of agents to select.

        Returns
        -------
        "Population[Agent]"
            List of selected agents of length n.
        """
        total_fitness = sum(population.fitness)
        if total_fitness == 0:
            return [random.choice(population) for _ in range(n)]
        distance = total_fitness / n
        start = random.uniform(0, distance)
        pointers = [start + i * distance for i in range(n)]
        selected = []
        current = 0
        for pointer in pointers:
            while pointer > 0:
                pointer -= population.fitness[current]
                current += 1
            selected.append(population[current - 1])
        return selected


class Truncation(Selection):
    """
    #! How is this different from Rank
    Selecting individuals based on truncation.

    Agents are sorted by their fitness, and only the top-performing fraction of
    the population is selected to reproduce.

    The selected individuals then produce the same number of offspring to fill
    the next generation.
    """

    @classmethod
    def select(cls, population: "Population[Agent]", n: int) -> "Population[Agent]":
        """
        Select n agents from the population.

        Parameters
        ----------
        population : "Population[Agent]"
            List of agents in the population, sorted desc by fitness.
        n : int
            Number of agents to select.

        Returns
        -------
        "Population[Agent]"
            List of selected agents of length n.
        """
        ranked_population = sorted(
            list(zip(population, population.fitness)), key=lambda x: x[1], reverse=True
        )
        selected = [agent for agent, _ in ranked_population[:n]]
        return selected


class Boltzmann(Selection):
    """
    Selecting individuals based on Boltzmann selection.

    The Boltzmann selection process involves selecting individuals based on their fitness
    and a temperature parameter.
    """

    @classmethod
    def select(
        cls, population: "Population[Agent]", n: int, temperature: float
    ) -> "Population[Agent]":
        """
        Select n agents from the population.

        Parameters
        ----------
        population : "Population[Agent]"
            List of agents in the population, sorted desc by fitness.
        n : int
            Number of agents to select.
        temperature : float
            Temperature parameter for Boltzmann selection.

        Returns
        -------
        "Population[Agent]"
            List of selected agents of length n.
        """
        probabilities = cls._boltzman_probabilities(population.fitness, temperature)
        selected = random.choices(population, probabilities, k=n)
        return selected

    def _boltzman_probabilities(self, fitnesses: list, temperature: float) -> list:
        """
        Calculate the selection probabilities based on Boltzman distribution.
        """
        exp_values = [
            self._boltzman_value(fitness, temperature) for fitness in fitnesses
        ]
        total_exp = sum(exp_values)
        probabilities = [exp / total_exp for exp in exp_values]
        return probabilities

    def _boltzman_value(self, fitness: float, temperature: float) -> float:
        """
        Calculate the Boltzman value for a given fitness.
        """
        return pow(2.71828, fitness / temperature)


class Softmax(Selection):
    """
    #! How is this different from Boltzmann
    Selecting individuals based on softmax selection.

    The softmax selection process involves selecting individuals based on their fitness
    and a temperature parameter.
    """

    @classmethod
    def select(
        cls, population: "Population[Agent]", n: int, temperature: float
    ) -> "Population[Agent]":
        """
        Select n agents from the population.

        Parameters
        ----------
        population : "Population[Agent]"
            List of agents in the population, sorted desc by fitness.
        n : int
            Number of agents to select.
        temperature : float
            Temperature parameter for softmax selection.

        Returns
        -------
        "Population[Agent]"
            List of selected agents of length n.
        """
        probabilities = cls._softmax_probabilities(population.fitness, temperature)
        selected = random.choices(population, probabilities, k=n)
        return selected

    def _softmax_probabilities(self, fitnesses: list, temperature: float) -> list:
        """
        Calculate the selection probabilities based on softmax distribution.
        """
        exp_values = [
            self._softmax_value(fitness, temperature) for fitness in fitnesses
        ]
        total_exp = sum(exp_values)
        probabilities = [exp / total_exp for exp in exp_values]
        return probabilities

    def _softmax_value(self, fitness: float, temperature: float) -> float:
        """
        Calculate the softmax value for a given fitness.
        """
        return pow(2.71828, fitness / temperature)


class FitnessScaling(Selection):
    """
    Selecting individuals based on fitness scaling.

    The fitness scaling process involves scaling the fitness values based on the
    average fitness of the population and then selecting individuals based on the
    scaled fitness values.

    Parameters
    ----------
    alpha : float
        Scaling factor for fitness scaling.
    """

    @classmethod
    def select(
        cls, population: "Population[Agent]", n: int, alpha: float
    ) -> "Population[Agent]":
        """
        Select n agents from the population.

        Parameters
        ----------
        population : "Population[Agent]"
            List of agents in the population, sorted desc by fitness.
        n : int
            Number of agents to select.
        alpha : float
            Scaling factor for fitness scaling.

        Returns
        -------
        "Population[Agent]"
            List of selected agents of length n.
        """
        scaled_fitnesses = cls._fitness_scaling(population.fitness, alpha)
        selected = random.choices(population, scaled_fitnesses, k=n)
        return selected

    def _fitness_scaling(self, fitnesses: list, alpha: float) -> list:
        """
        Scale the fitness values based on the average fitness of the population.
        """
        average_fitness = sum(fitnesses) / len(fitnesses)
        scaled_fitnesses = [1 + alpha * (f - average_fitness) for f in fitnesses]
        return scaled_fitnesses
