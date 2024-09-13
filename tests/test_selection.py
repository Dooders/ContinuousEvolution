import pytest
from evolution.selection import RouletteWheel, Tournament, Rank, Elitism, StochasticUniversalSampling, Truncation, Boltzmann, Softmax, FitnessScaling
from evolution.population import Population
from evolution.agent import Agent

class DummyAgent(Agent):
    def __init__(self, fitness):
        self.fitness_value = fitness

    @property
    def fitness(self):
        return self.fitness_value

@pytest.fixture
def population():
    return Population([
        DummyAgent(0.5),
        DummyAgent(0.7),
        DummyAgent(0.3),
        DummyAgent(0.6),
        DummyAgent(0.4)
    ])

def test_roulette_wheel_selection(population):
    selected = RouletteWheel.select(population, 2)
    assert len(selected) == 2
    assert all(isinstance(agent, Agent) for agent in selected)

def test_tournament_selection(population):
    selected = Tournament.select(population, 2, k=3)
    assert len(selected) == 2
    assert all(isinstance(agent, Agent) for agent in selected)

def test_rank_selection(population):
    selected = Rank.select(list(zip(population, [agent.fitness for agent in population])), 2)
    assert len(selected) == 2
    assert all(isinstance(agent, Agent) for agent in selected)

def test_elitism_selection(population):
    selected = Elitism.select(population, 2)
    assert len(selected) == 2
    assert selected[0].fitness == 0.7
    assert selected[1].fitness == 0.6

def test_stochastic_universal_sampling(population):
    selected = StochasticUniversalSampling.select(population, 2)
    assert len(selected) == 2
    assert all(isinstance(agent, Agent) for agent in selected)

def test_truncation_selection(population):
    selected = Truncation.select(population, 2)
    assert len(selected) == 2
    assert selected[0].fitness == 0.7
    assert selected[1].fitness == 0.6

def test_boltzmann_selection(population):
    selected = Boltzmann.select(population, 2, temperature=1.0)
    assert len(selected) == 2
    assert all(isinstance(agent, Agent) for agent in selected)

def test_softmax_selection(population):
    selected = Softmax.select(population, 2, temperature=1.0)
    assert len(selected) == 2
    assert all(isinstance(agent, Agent) for agent in selected)

def test_fitness_scaling_selection(population):
    selected = FitnessScaling.select(population, 2, alpha=1.0)
    assert len(selected) == 2
    assert all(isinstance(agent, Agent) for agent in selected)