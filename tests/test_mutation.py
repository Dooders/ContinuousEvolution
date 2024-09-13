import pytest
import torch
import torch.nn as nn
from evolution.agent import Agent
from evolution.mutation import Uniform, Gaussian, NonUniform, BitFlip, Boundary, AdaptiveGaussian, Hybrid

class SimpleModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        return self.linear(x)

@pytest.fixture
def agent():
    model = SimpleModel(1, 3)
    return Agent(model, arguments={"input_dim": 1, "hidden_dim": 3})

def test_uniform_mutation(agent):
    mutation = Uniform()
    mutated_agent = mutation.mutate(agent)
    assert isinstance(mutated_agent, Agent)

def test_gaussian_mutation(agent):
    mutation = Gaussian()
    mutated_agent = mutation.mutate(agent)
    assert isinstance(mutated_agent, Agent)

def test_non_uniform_mutation(agent):
    mutation = NonUniform()
    mutated_agent = mutation.mutate(agent)
    assert isinstance(mutated_agent, Agent)

def test_bit_flip_mutation(agent):
    mutation = BitFlip()
    mutated_agent = mutation.mutate(agent)
    assert isinstance(mutated_agent, Agent)

def test_boundary_mutation(agent):
    mutation = Boundary()
    mutated_agent = mutation.mutate(agent)
    assert isinstance(mutated_agent, Agent)

def test_adaptive_gaussian_mutation(agent):
    mutation = AdaptiveGaussian()
    mutated_agent = mutation.mutate(agent)
    assert isinstance(mutated_agent, Agent)

def test_hybrid_mutation(agent):
    mutation = Hybrid()
    mutated_agent = mutation.mutate(agent)
    assert isinstance(mutated_agent, Agent)