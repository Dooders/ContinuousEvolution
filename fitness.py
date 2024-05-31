from abc import abstractmethod
from typing import List

from agent import Agent
from population import Population


class Fitness:
    """
    Abstract class for fitness functions.
    """

    @classmethod
    @abstractmethod
    def evaluate(cls, population: Population[Agent], results: list) -> list:

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

    def evaluate(self, population, results: list) -> list:
        pass
