from abc import ABC, abstractmethod

from agent import Agent
from population import Population


class Fitness(ABC):
    """
    Abstract class for fitness functions.

    Intended to be used as a base class to be inherited where the evaluate method
    returns a Population object of the agent updated fitness scores

    A Fitness subclass is passed to the ContinuousEvolution object during initialization.

    Methods
    -------
    score(population: Population[Agent], results: list) -> Population[Agent]:
        Evaluate the fitness of the agents in the population.
    """

    @classmethod
    @abstractmethod
    def score(
        cls, population: "Population[Agent]", results: list
    ) -> "Population[Agent]":
        """
        Classmethod to evaluate the fitness of the agents in the population.

        Parameters
        ----------
        population : Population[Agent]
            The population of agents to evaluate.
        results : list
            The results of the agents' actions.

        Returns
        -------
        Population[Agent]
            The updated population of agents with updated fitness scores.
        """

        raise NotImplementedError("Subclasses must implement this method")


class TradingFitness(Fitness):
    """
    Will return if the agent should be removed or not based on the fitness value.

    Need to figure out how to score the agent based on the outcome.

    Fitness scenarios:
    - Agent sells, loses money (removed)
    - Agent sells, makes money (remains)
    - Agent does not sell, misses out on  R profit (removed)
    - Agent does not sell, avoids loss (remains)
    - Agent sells, has loss but has a good history (remains?)
    - Agent has N streak of losses, even with net profit (removed)

    N is the number of losses in a row before the agent is removed.
    R is the profit that the agent missed out on.

    So the class will need the agent's history and the results from the agent's actions.
    """

    def score(
        self, population: "Population[Agent]", results: list
    ) -> "Population[Agent]":
        pass
