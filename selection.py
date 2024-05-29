import random
from abc import ABC, abstractmethod

from agent import Agent


class Selection(ABC):
    @abstractmethod
    def select(self, population: list, fitnesses: list) -> "Agent":
        """Select an agent from the population based on fitnesses."""
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
        """
        Select n agents from the population based on fitnesses.

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
    population : list
        List of agents in the population.
    fitnesses : list
        List of fitness values corresponding to each agent in the population.
    k : int
        Number of individuals to select in each tournament.
    """

    def __init__(self, population: list, fitnesses: list, k: int) -> None:
        self.population = population
        self.fitnesses = fitnesses
        self.k = k

    def select(self, n: int) -> list:
        """
        Select n agents from the population based on fitnesses.

        Parameters
        ----------
        n : int
            Number of agents to select.

        Returns
        -------
        list
            List of selected agents.
        """
        selected = []
        for _ in range(n):
            tournament = random.sample(
                list(zip(self.population, self.fitnesses)), self.k
            )
            winner = max(tournament, key=lambda x: x[1])[0]
            selected.append(winner)
        return selected


class Rank(Selection):
    """
    Selecting individuals based on their rank in the population.

    The rank selection process involves sorting the population based on fitness
    and then selecting individuals based on their rank.

    Parameters
    ----------
    population : list
        List of agents in the population.
    fitnesses : list
        List of fitness values corresponding to each agent in the population.
    """

    def __init__(self, population: list, fitnesses: list) -> None:
        self.population = population
        self.fitnesses = fitnesses

    def select(self, n: int) -> list:
        """
        Select n agents from the population based on fitnesses.

        Parameters
        ----------
        n : int
            Number of agents to select.

        Returns
        -------
        list
            List of selected agents.
        """
        ranked_population = sorted(
            list(zip(self.population, self.fitnesses)), key=lambda x: x[1], reverse=True
        )
        selected = [agent for agent, _ in ranked_population[:n]]
        return selected


class Elitism(Selection):
    """
    Selecting individuals based on elitism.

    The elitism selection process involves selecting the best individuals from the population.

    Parameters
    ----------
    population : list
        List of agents in the population.
    fitnesses : list
        List of fitness values corresponding to each agent in the population.
    """

    def __init__(self, population: list, fitnesses: list) -> None:
        self.population = population
        self.fitnesses = fitnesses

    def select(self, n: int) -> list:
        """
        Select n agents from the population based on fitnesses.

        Parameters
        ----------
        n : int
            Number of agents to select.

        Returns
        -------
        list
            List of selected agents.
        """
        ranked_population = sorted(
            list(zip(self.population, self.fitnesses)), key=lambda x: x[1], reverse=True
        )
        selected = [agent for agent, _ in ranked_population[:n]]
        return selected


class StochasticUniversalSampling(Selection):
    """
    Selecting individuals based on stochastic universal sampling.

    The stochastic universal sampling selection process involves selecting individuals
    based on their fitness and a random starting point.

    Parameters
    ----------
    population : list
        List of agents in the population.
    fitnesses : list
        List of fitness values corresponding to each agent in the population.
    """

    def __init__(self, population: list, fitnesses: list) -> None:
        self.population = population
        self.fitnesses = fitnesses

    def select(self, n: int) -> list:
        """
        Select n agents from the population based on fitnesses.

        Parameters
        ----------
        n : int
            Number of agents to select.

        Returns
        -------
        list
            List of selected agents.
        """
        total_fitness = sum(self.fitnesses)
        if total_fitness == 0:
            return [random.choice(self.population) for _ in range(n)]
        distance = total_fitness / n
        start = random.uniform(0, distance)
        pointers = [start + i * distance for i in range(n)]
        selected = []
        current = 0
        for pointer in pointers:
            while pointer > 0:
                pointer -= self.fitnesses[current]
                current += 1
            selected.append(self.population[current - 1])
        return selected


class Truncation(Selection):
    """
    Selecting individuals based on truncation.

    Agents are sorted by their fitness, and only the top-performing fraction of
    the population is selected to reproduce.

    The selected individuals then produce the same number of offspring to fill
    the next generation.

    Parameters
    ----------
    population : list
        List of agents in the population.
    fitnesses : list
        List of fitness values corresponding to each agent in the population.
    """

    def __init__(self, population: list, fitnesses: list) -> None:
        self.population = population
        self.fitnesses = fitnesses

    def select(self, n: int) -> list:
        """
        Select n agents from the population based on fitnesses.

        Parameters
        ----------
        n : int
            Number of agents to select.

        Returns
        -------
        list
            List of selected agents.
        """
        ranked_population = sorted(
            list(zip(self.population, self.fitnesses)), key=lambda x: x[1], reverse=True
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
    population : list
        List of agents in the population.
    fitnesses : list
        List of fitness values corresponding to each agent in the population.
    temperature : float
        Temperature parameter for Boltzman selection.
    """

    def __init__(self, population: list, fitnesses: list, temperature: float) -> None:
        self.population = population
        self.fitnesses = fitnesses
        self.temperature = temperature

    def select(self, n: int) -> list:
        """
        Select n agents from the population based on fitnesses.

        Parameters
        ----------
        n : int
            Number of agents to select.

        Returns
        -------
        list
            List of selected agents.
        """
        probabilities = self._boltzman_probabilities()
        selected = random.choices(self.population, probabilities, k=n)
        return selected

    def _boltzman_probabilities(self) -> list:
        """
        Calculate the selection probabilities based on Boltzman distribution.

        Returns
        -------
        list
            List of selection probabilities.
        """
        exp_values = [self._boltzman_value(fitness) for fitness in self.fitnesses]
        total_exp = sum(exp_values)
        probabilities = [exp / total_exp for exp in exp_values]
        return probabilities

    def _boltzman_value(self, fitness: float) -> float:
        """
        Calculate the Boltzman value for a given fitness.

        Parameters
        ----------
        fitness : float
            Fitness value of an agent.

        Returns
        -------
        float
            Boltzman value.
        """
        return pow(2.71828, fitness / self.temperature)


class Softmax(Selection):
    """
    Selecting individuals based on softmax selection.

    The softmax selection process involves selecting individuals based on their fitness
    and a temperature parameter.

    Parameters
    ----------
    population : list
        List of agents in the population.
    fitnesses : list
        List of fitness values corresponding to each agent in the population.
    temperature : float
        Temperature parameter for softmax selection.
    """

    def __init__(self, population: list, fitnesses: list, temperature: float) -> None:
        self.population = population
        self.fitnesses = fitnesses
        self.temperature = temperature

    def select(self, n: int) -> list:
        """
        Select n agents from the population based on fitnesses.

        Parameters
        ----------
        n : int
            Number of agents to select.

        Returns
        -------
        list
            List of selected agents.
        """
        probabilities = self._softmax_probabilities()
        selected = random.choices(self.population, probabilities, k=n)
        return selected

    def _softmax_probabilities(self) -> list:
        """
        Calculate the selection probabilities based on softmax distribution.

        Returns
        -------
        list
            List of selection probabilities.
        """
        exp_values = [self._softmax_value(fitness) for fitness in self.fitnesses]
        total_exp = sum(exp_values)
        probabilities = [exp / total_exp for exp in exp_values]
        return probabilities

    def _softmax_value(self, fitness: float) -> float:
        """
        Calculate the softmax value for a given fitness.

        Parameters
        ----------
        fitness : float
            Fitness value of an agent.

        Returns
        -------
        float
            Softmax value.
        """
        return pow(2.71828, fitness / self.temperature)


class FitnessScaling(Selection):
    """
    Selecting individuals based on fitness scaling.

    The fitness scaling process involves scaling the fitness values based on the
    average fitness of the population and then selecting individuals based on the
    scaled fitness values.

    Parameters
    ----------
    population : list
        List of agents in the population.
    fitnesses : list
        List of fitness values corresponding to each agent in the population.
    alpha : float
        Scaling factor for fitness scaling.
    """

    def __init__(self, population: list, fitnesses: list, alpha: float) -> None:
        self.population = population
        self.fitnesses = fitnesses
        self.alpha = alpha

    def select(self, n: int) -> list:
        """
        Select n agents from the population based on fitnesses.

        Parameters
        ----------
        n : int
            Number of agents to select.

        Returns
        -------
        list
            List of selected agents.
        """
        scaled_fitnesses = self._fitness_scaling()
        selected = random.choices(self.population, scaled_fitnesses, k=n)
        return selected

    def _fitness_scaling(self) -> list:
        """
        Scale the fitness values based on the average fitness of the population.

        Returns
        -------
        list
            List of scaled fitness values.
        """
        average_fitness = sum(self.fitnesses) / len(self.fitnesses)
        scaled_fitnesses = [
            1 + self.alpha * (f - average_fitness) for f in self.fitnesses
        ]
        return scaled_fitnesses


#! TODO: Implement Lexicase selection
