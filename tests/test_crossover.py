import pytest
import torch
import torch.nn as nn
from evolution.agent import Agent
from evolution.crossover import Average, Random, RandomPoint, RandomRange

class SimpleModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        return self.linear(x)

@pytest.fixture
def parent_x():
    model = SimpleModel(1, 3)
    return Agent(model, arguments={"input_dim": 1, "hidden_dim": 3})

@pytest.fixture
def parent_y():
    model = SimpleModel(1, 3)
    return Agent(model, arguments={"input_dim": 1, "hidden_dim": 3})

def test_average_crossover(parent_x, parent_y):
    crossover = Average()
    child = crossover.crossover(parent_x, parent_y)
    assert isinstance(child, Agent)
    assert child.arguments == parent_x.arguments

def test_random_crossover(parent_x, parent_y):
    crossover = Random()
    child = crossover.crossover(parent_x, parent_y)
    assert isinstance(child, Agent)
    assert child.arguments == parent_x.arguments

def test_random_point_crossover(parent_x, parent_y):
    crossover = RandomPoint()
    child = crossover.crossover(parent_x, parent_y)
    assert isinstance(child, Agent)
    assert child.arguments == parent_x.arguments

def test_random_range_crossover(parent_x, parent_y):
    crossover = RandomRange()
    child = crossover.crossover(parent_x, parent_y)
    assert isinstance(child, Agent)
    assert child.arguments == parent_x.arguments