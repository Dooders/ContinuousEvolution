import random
from abc import ABC, abstractmethod

from agent import Agent


class Selection(ABC):
    @abstractmethod
    def select(self, population: list, fitnesses: list, n: int) -> "Agent":
        """
        Select an agent from the population based on fitnesses.

        Parameters
        ----------
        population : list
            List of agents in the population.
        fitnesses : list
            List of fitness values corresponding to each agent in the population.
        n : int
            Number of agents to select.

        Returns
        -------
        list
            List of selected agents.
        """
        raise NotImplementedError("Select method not implemented.")


class RouletteWheel(Selection):
    """
    Selecting individuals based on their fitness relative to the population.

    The probability of an individual being selected is proportional to its fitness.

    This simulates a roulette wheel where each individual occupies a slice of the
    wheel proportional to its fitness score.
    """

    @classmethod
    def select(cls, population: list, fitnesses: list, n: int) -> list:
        total_fitness = sum(fitnesses)
        # Avoid division by zero; randomly choose an agent
        if total_fitness == 0:
            return [random.choice(population) for _ in range(n)]
        probabilities = [f / total_fitness for f in fitnesses]
        selected = random.choices(population, probabilities, k=n)
        return selected


class Tournament(Selection):
    """
    Selecting individuals based on a tournament selection process.

    The tournament selection process involves selecting k individuals from the population
    and then selecting the best individual from the k individuals.

    Parameters
    ----------
    k : int
        Number of individuals to select in each tournament.
    """

    @classmethod
    def select(cls, population: list, fitnesses: list, k: int, n: int) -> list:
        selected = []
        for _ in range(n):
            tournament = random.sample(list(zip(population, fitnesses)), k)
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
    def select(cls, population: list, fitnesses: list, n: int) -> list:
        ranked_population = sorted(
            list(zip(population, fitnesses)), key=lambda x: x[1], reverse=True
        )
        selected = [agent for agent, _ in ranked_population[:n]]
        return selected


class Elitism(Selection):
    """
    #! How is this different from Rank
    Selecting individuals based on elitism.

    The elitism selection process involves selecting the best individuals from the population.
    """

    @classmethod
    def select(cls, population: list, fitnesses: list, n: int) -> list:
        ranked_population = sorted(
            list(zip(population, fitnesses)), key=lambda x: x[1], reverse=True
        )
        selected = [agent for agent, _ in ranked_population[:n]]
        return selected


class StochasticUniversalSampling(Selection):
    """
    Selecting individuals based on stochastic universal sampling.

    The stochastic universal sampling selection process involves selecting individuals
    based on their fitness and a random starting point.
    """

    def select(cls, population: list, fitnesses: list, n: int) -> list:
        total_fitness = sum(fitnesses)
        if total_fitness == 0:
            return [random.choice(population) for _ in range(n)]
        distance = total_fitness / n
        start = random.uniform(0, distance)
        pointers = [start + i * distance for i in range(n)]
        selected = []
        current = 0
        for pointer in pointers:
            while pointer > 0:
                pointer -= fitnesses[current]
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
    def select(cls, population: list, fitnesses: list, n: int) -> list:
        ranked_population = sorted(
            list(zip(population, fitnesses)), key=lambda x: x[1], reverse=True
        )
        selected = [agent for agent, _ in ranked_population[:n]]
        return selected


class Boltzmann(Selection):
    """
    Selecting individuals based on Boltzmann selection.

    The Boltzmann selection process involves selecting individuals based on their fitness
    and a temperature parameter.

    Parameters
    ----------
    temperature : float
        Temperature parameter for Boltzman selection.
    """

    @classmethod
    def select(
        cls, population: list, fitnesses: list, n: int, temperature: float
    ) -> list:
        probabilities = cls._boltzman_probabilities(fitnesses, temperature)
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

    Parameters
    ----------
    temperature : float
        Temperature parameter for softmax selection.
    """

    @classmethod
    def select(
        cls, population: list, fitnesses: list, n: int, temperature: float
    ) -> list:
        probabilities = cls._softmax_probabilities(fitnesses, temperature)
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
    def select(cls, population: list, fitnesses: list, n: int, alpha: float) -> list:
        scaled_fitnesses = cls._fitness_scaling(fitnesses, alpha)
        selected = random.choices(population, scaled_fitnesses, k=n)
        return selected

    def _fitness_scaling(self, fitnesses: list, alpha: float) -> list:
        """
        Scale the fitness values based on the average fitness of the population.
        """
        average_fitness = sum(fitnesses) / len(fitnesses)
        scaled_fitnesses = [1 + alpha * (f - average_fitness) for f in fitnesses]
        return scaled_fitnesses
