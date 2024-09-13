import pytest
import torch
import torch.nn as nn
from evolution.evolution import ContinuousEvolution
from evolution.fitness import Fitness
from evolution.crossover import Average
from evolution.mutation import Gaussian

class SimpleModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        return self.linear(x)

class DummyFitness(Fitness):
    @classmethod
    def score(cls, population, results):
        return population

@pytest.fixture
def continuous_evolution():
    return ContinuousEvolution(
        agent_type=SimpleModel,
        fitness=DummyFitness(),
        settings={"input_dim": 1, "hidden_dim": 3},
        population_size=10,
        parent_count=2,
        crossover_strategy=Average(),
        mutation_strategy=Gaussian(),
    )

def test_continuous_evolution_initialization(continuous_evolution):
    assert isinstance(continuous_evolution, ContinuousEvolution)
    assert len(continuous_evolution.population) == 10

def test_continuous_evolution_run(continuous_evolution):
    X = torch.randn(1)
    output = continuous_evolution.run(X, sequence_length=5)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 1)

def test_continuous_evolution_simulate(continuous_evolution):
    model = SimpleModel(1, 3)
    results = continuous_evolution.simulate(model, starting_value=0.5, cycles=10)
    assert len(results) == 10
    assert all(isinstance(r, float) for r in results)