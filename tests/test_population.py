import pytest
from evolution.population import Population, CollectiveState
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
        DummyAgent(0.3)
    ])

def test_population_sort_by_fitness(population):
    sorted_population = population.sort_by_fitness()
    assert [agent.fitness for agent in sorted_population] == [0.7, 0.5, 0.3]

def test_population_sorted(population):
    sorted_list = population.sorted()
    assert [fitness for _, fitness in sorted_list] == [0.7, 0.5, 0.3]

def test_collective_state():
    state = CollectiveState()
    agent1 = DummyAgent(0.5)
    agent2 = DummyAgent(0.7)

    state.add_agent(agent1)
    state.add_agent(agent2)
    state.add_relationship(agent1.id, agent2.id, "parent")

    assert len(state.graph.nodes) == 2
    assert len(state.graph.edges) == 2  # Bidirectional edge

    state.save_state(agent1.id, {"generation": 1})
    assert len(state.active_states) == 1