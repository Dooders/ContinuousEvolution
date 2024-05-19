class Populations(list):
    def __init__(self, population_size: int):
        self.population_size = population_size

    def select(self, selection_strategy):
        return selection_strategy(self)

    def reproduce(self, reproduction_strategy):
        # Select the parents
        # Create offspring
        # Mutate the offspring
        pass

    def vote(self, voting_strategy):
        return voting_strategy(self)

    def fitness(self, fitness_strategy):
        return fitness_strategy(self)
