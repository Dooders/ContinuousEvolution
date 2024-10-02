from collections import deque
from typing import List

import numpy as np
import torch
import torch.nn as nn

from evolution.agent import Agent
from evolution.crossover import Average
from evolution.fitness import Fitness
from evolution.models.simple import SimpleSequentialNetwork
from evolution.mutation import Gaussian
from evolution.selection import Rank


class ContinuousEvolution:
    """
    Class to perform continuous evolution on a population of neural networks.

    Parameters
    ----------
    agent_model : nn.Module
        Neural network model to evolve.
    criterion : nn.Module
        Loss function to use.
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
    agent_model : nn.Module
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
    run(X: torch.Tensor, sequence_length: int) -> torch.Tensor:
        Run the evolutionary training process. The sequence length is the number of past
        inputs to consider for each prediction.
    """

    def __init__(self, params: dict) -> None:
        self.agent_type = params["agent_model"]
        self.settings = params["settings"]
        self.population_size = params["population_size"]
        self.population = self._initialize_population()
        self.parent_count = params["num_parents"]
        self.criterion = params["fitness"]
        self.population_history = []
        self.predictions = []
        self.fitness = params["fitness"]
        self.crossover_strategy = params["crossover_strategy"]
        self.mutation_strategy = params["mutation_strategy"]
        self.target_buffer = deque(maxlen=params["target_buffer_size"])
        self.output_buffer = deque(maxlen=params["output_buffer_size"])

    def _initialize_population(self) -> list:
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
        return [
            Agent(self.agent_type, self.settings) for _ in range(self.population_size)
        ]

    def evaluate_agents(
        self, population: List[torch.Module], price_history: List[float]
    ) -> list[float]:
        """
        Evaluate the fitness of the agents.

        If the price went down, the agent is rewarded for the output being close to 1.
        If the price went up, the agent is rewarded for the output being close to 0.

        Parameters
        ----------
        outputs : list[float]
            The outputs of the agents.
        price_went_down : bool
            Whether the price went down.

        Returns
        -------
        list[float]
            The fitness of the agents.
        """
        scores = []
        for agent in population:
            scores.append(agent.evaluate(self.criterion, price_history))
        return scores

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
        return self.crossover_strategy.crossover(
            parent1, parent2, self.agent_type, self.settings
        )

    def mutate(self, agent: Agent) -> None:
        """Mutate the given agent."""
        self.mutation_strategy.mutate(agent.model, 9.9)
        agent.update_id()  # Update the ID after mutation

    def run(
        self,
        X: torch.Tensor,
        sequence_length: int,
    ) -> torch.Tensor:
        """
        Run the evolutionary training process.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor to predict from.
        sequence_length : int
            Length of the sequence to consider for each prediction.

        Returns
        -------
        torch.Tensor
            Predicted output tensor from the best model in the population.
        """

        self.target_buffer.append(X)
        sequence_list = list(self.target_buffer)[-sequence_length:]
        input_sequence = torch.tensor(sequence_list, dtype=torch.float32).unsqueeze(
            -1
        )  # Add feature dimension
        best_model = self.population[0]

        # Evaluate the fitness of the population and select parents
        # This is a delayed feedback loop from previous predictions
        if len(self.output_buffer) > 1:
            past_outputs = self.output_buffer[-1]
            actual_target = self.target_buffer[-1]
            past_actual_target = self.target_buffer[-2]
            fitnesses = self.fitness.evaluate(
                past_outputs, actual_target, past_actual_target
            )
            best_model_index = self.fitness.most_fit
            best_model = self.population[best_model_index]
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
                f"Average fitness: {np.mean(fitnesses):.4f} Best: {max(fitnesses):.4f} Worst: {min(fitnesses):.4f}"
            )
        with torch.no_grad():
            new_outputs = [model(input_sequence) for model in self.population]
            self.output_buffer.append(new_outputs)

            #! This is a hack to get the best model from the population
            #! This should be done in a better way
            final_output = best_model(input_sequence)
            self.predictions.append(final_output)

            return final_output

    def simulate(
        self, model: nn.Module, starting_value: float, cycles: int = 100
    ) -> list:
        """
        Simulate the model for a number of cycles to assess its performance
        without reproduction or fitness evaluation.

        Parameters
        ----------
        model : nn.Module
            Model to simulate.
        starting_value : float
            Starting value for the simulation.
        cycles : int
            Number of cycles to simulate.

        Returns
        -------
        list
            List of results from the simulation.
        """
        previous = starting_value
        results = []
        for _ in range(cycles):
            value = torch.tensor(previous, dtype=torch.float32)
            prediction = model(value.unsqueeze(0).unsqueeze(-1))
            results.append(prediction.item())
            previous = prediction.item()

        return results
