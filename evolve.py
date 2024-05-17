from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from agent import Agent
from fitness import Fitness


class ContinuousEvolution:
    """
    Class to perform continuous evolution on a population of neural networks.

    Parameters
    ----------
    model : nn.Module
        Neural network model to evolve.
    settings : dict
        Settings to pass to the model factory.
    population_size : int
        Number of networks in the population.
    parent_count : int
        Number of parents to select from the population.
    crossover_strategy : CrossoverStrategy, optional
        Crossover strategy to use, by default AverageCrossover.
    mutation_strategy : MutationStrategy, optional
        Mutation strategy to use, by default GaussianMutation.

    Attributes
    ----------
    model : nn.Module
        Neural network model to evolve.
    criterion : nn.Module
        Loss function to use.
    population_size : int
        Number of networks in the population.
    population : list
        List of neural networks in the population.
    parent_count : int
        Number of parents to select from the population.
    population_history : list
        List of populations at each cycle.
    fitness_history : list
        List of fitness values at each cycle. Tuple of (min, max, avg).
    crossover_strategy : CrossoverStrategy
        Crossover strategy to use.
    mutation_strategy : MutationStrategy
        Mutation strategy to use.

    Methods
    -------
    select_parents(population: list, fitnesses: list, num_parents: int) -> list:
        Select the best parents from the population based on fitness.
    crossover(parent1: nn.Module, parent2: nn.Module) -> nn.Module:
        Create a child network by averaging the weights of two parent networks.
    mutate(network: nn.Module) -> None:
        Mutate the weights of a neural network in place.
    run(cycles: int, X: torch.Tensor, y: torch.Tensor, history: bool = True) -> tuple:
        Run the evolutionary training process.
    log_fitness(fitness: list) -> Tuple[float, float, float]:
        Calculates the Minimum, Maximum and Average fitness values for each generation.
    """

    def __init__(
        self,
        model: nn.Module,
        settings: dict,
        population_size: int,
        parent_count: int,
        crossover_strategy,
        mutation_strategy,
    ) -> None:
        self.model = model
        self.settings = settings
        self.criterion = nn.MSELoss()
        self.population_size = population_size
        self.population = self._initialize_population(population_size)
        self.parent_count = parent_count
        self.population_history = []
        self.fitness = Fitness(self.criterion)
        self.crossover_strategy = crossover_strategy
        self.mutation_strategy = mutation_strategy

    def _initialize_population(self, size: int) -> list:
        """
        Initialize a population of neural networks.

        Parameters
        ----------
        size : int
            Number of networks in the population.

        Returns
        -------
        list
            List of SimpleSequentialNetwork instances.
        """
        return [Agent(self.model(**self.settings)) for _ in range(size)]

    def select_parents(
        self, population: list, fitnesses: list, parent_count: int
    ) -> list:
        """
        Select the best parents from the population based on fitness.

        Parameters
        ----------
        population : list
            List of neural networks.
        fitnesses : list
            List of fitness values corresponding to each network in the population.
        parent_count : int
            Number of parents to select.

        Returns
        -------
        list
            List of selected parents (neural networks).
        """
        parents = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)
        return [parent for parent, _ in parents[:parent_count]]

    def crossover(self, parent1: nn.Module, parent2: nn.Module) -> nn.Module:
        """
        Create a child network by the crossover of two parent networks.

        Parameters
        ----------
        parent1 : nn.Module
            First parent network.
        parent2 : nn.Module
            Second parent network.

        Returns
        -------
        nn.Module
            Child network created by crossover of the parents.
        """
        return self.crossover_strategy.crossover(parent1, parent2)

    def mutate(self, network: nn.Module) -> None:
        """
        Mutate the weights of a neural network in place.

        Parameters
        ----------
        network : nn.Module
            Neural network to mutate.
        """
        self.mutation_strategy.mutate(network)

    def run(
        self,
        max_cycles: int,
        X: torch.Tensor,
        y: torch.Tensor,
        history: bool = True,
        early_stopping: float = None,
    ) -> tuple:
        """
        Run the evolutionary training process.

        First, evaluate the fitness of each network in the population.
        Then, select the best parents based on fitness.
        Create a new generation by crossing over and mutating the parents.
        Repeat for the specified number of cycles.

        Parameters
        ----------
        max_cycles : int
            Max number of training cycles.
        X : torch.Tensor
            Input data of shape (batch_size, num_features).
        y : torch.Tensor
            Target labels of shape (batch_size, 1).
        history : bool, optional
            Whether to store the population history, by default True.
        early_stopping : float, optional
            Stop training if the best fitness exceeds this value, by default None.

        Returns
        -------
        tuple
            Final population of neural networks and the best fit network.
        """

        for cycle in range(max_cycles):
            if history:
                self.population_history.append(self.population.copy())
            fitnesses = [
                self.evaluate_fitness(net, self.criterion, X, y)
                for net in self.population
            ]
            self.fitness_history.append(self.log_fitness(fitnesses))
            parents = self.select_parents(self.population, fitnesses, self.parent_count)

            next_generation = []
            while len(next_generation) < self.population_size:
                for i in range(len(parents)):
                    for j in range(i + 1, len(parents)):
                        child = self.crossover(parents[i], parents[j])
                        self.mutate(child)
                        next_generation.append(child)
                        if len(next_generation) >= self.population_size:
                            break
            self.population = next_generation

            print(
                f"Cycle {cycle}: Average fitness: {np.mean(fitnesses):.2f} Best: {max(fitnesses):.2f} Worst: {min(fitnesses):.2f}"
            )
            if early_stopping and max(fitnesses) >= -early_stopping:
                print(f"Early stopping at cycle {cycle}")
                break

        if history:
            self.population_history.append(self.population.copy())

        return self.population, self.best(X, y)

    def log_fitness(self, fitness: list) -> Tuple[float, float, float]:
        """
        Stores the Minimum, Maximum and Average fitness values for each generation.

        Parameters
        ----------
        fitness : list
            List of fitness values for each network in the population.

        Returns
        -------
        Tuple[float, float, float]
            Minimum, Maximum and Average fitness values.
        """
        return min(fitness), max(fitness), np.mean(fitness)
