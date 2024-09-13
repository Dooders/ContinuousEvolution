import pytest
import torch
from evolution.loss import DirectionalMSELoss

@pytest.fixture
def directional_mse_loss():
    return DirectionalMSELoss(weight=0.5)

def test_directional_mse_loss(directional_mse_loss):
    predictions = torch.tensor([1.0, 2.0, 3.0])
    targets = torch.tensor([1.5, 1.8, 3.2])
    previous_values = torch.tensor([0.8, 2.1, 2.9])

    loss = directional_mse_loss(predictions, targets, previous_values)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Scalar tensor

def test_directional_mse_loss_direction(directional_mse_loss):
    predictions = torch.tensor([1.0, 2.0, 3.0])
    targets = torch.tensor([1.5, 1.8, 3.2])
    previous_values = torch.tensor([0.8, 2.1, 2.9])

    loss1 = directional_mse_loss(predictions, targets, previous_values)

    # Reverse the direction of predictions
    reversed_predictions = torch.tensor([0.6, 2.2, 2.8])
    loss2 = directional_mse_loss(reversed_predictions, targets, previous_values)

    assert loss2 > loss1  # The loss should be higher when directions are incorrect