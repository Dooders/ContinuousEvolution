from evolution.agent import Agent
from evolution.crossover import CrossoverStrategy
from evolution.mutation import MutationStrategy

#! need ReproductionStrategy approach???

#! maybe you give the agent everything it needs to "survive" and let it go, see how it does, and then you can evaluate it based on its performance


class Reproduction:
    def __init__(
        self,
        population_size: int,
        mutation: "MutationStrategy",
        crossover: "CrossoverStrategy",
    ):
        self.population_size = population_size
        self.mutation = mutation
        self.crossover = crossover

    def create(self, parent_x: "Agent", parent_y: "Agent") -> "Agent":
        # Crossover
        # Mutation
        # Any other processing
        pass
