import pytest
import torch
import torch.nn as nn
import numpy as np
from evolution.utils import model_hash, extract_parameters, set_seed

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 3),
            nn.ReLU(),
            nn.Linear(3, 1)
        )

    def forward(self, x):
        return self.net(x)

@pytest.fixture
def simple_model():
    return SimpleModel()

def test_model_hash(simple_model):
    hash1 = model_hash(simple_model)
    hash2 = model_hash(simple_model)
    assert isinstance(hash1, str)
    assert hash1 == hash2

    # Change model parameters
    for param in simple_model.parameters():
        param.data += 0.1
    
    hash3 = model_hash(simple_model)
    assert hash1 != hash3

def test_extract_parameters(simple_model):
    params = extract_parameters(simple_model)
    assert isinstance(params, np.ndarray)
    
    # Calculate expected number of parameters
    expected_params = 2 * 3 + 3 + 3 * 1 + 1  # weights and biases for two linear layers
    assert params.shape == (expected_params,)

def test_set_seed_int():
    set_seed(42)
    rand1 = torch.rand(1)
    np_rand1 = np.random.rand()

    set_seed(42)
    rand2 = torch.rand(1)
    np_rand2 = np.random.rand()

    assert torch.allclose(rand1, rand2)
    assert np_rand1 == np_rand2

def test_set_seed_str():
    set_seed("hello")
    rand1 = torch.rand(1)
    np_rand1 = np.random.rand()

    set_seed("hello")
    rand2 = torch.rand(1)
    np_rand2 = np.random.rand()

    assert torch.allclose(rand1, rand2)
    assert np_rand1 == np_rand2

def test_set_seed_different():
    set_seed(42)
    rand1 = torch.rand(1)
    np_rand1 = np.random.rand()

    set_seed(43)
    rand2 = torch.rand(1)
    np_rand2 = np.random.rand()

    assert not torch.allclose(rand1, rand2)
    assert np_rand1 != np_rand2