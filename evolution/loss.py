import torch
import torch.nn as nn


class DirectionalMSELoss(nn.Module):
    """
    Directional MSE Loss.

    Parameters
    ----------
    weight : float
        Weight for the direction loss. Default is 0.1.

    Methods
    -------
    forward(predictions: torch.Tensor, targets: torch.Tensor, previous_values: torch.Tensor) -> torch.Tensor:
        Calculate the Directional MSE Loss.
    """

    def __init__(self, weight: float = 0.5) -> None:
        super(DirectionalMSELoss, self).__init__()
        self.weight = weight
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        previous_values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate the Directional MSE Loss.

        Parameters
        ----------
        predictions : torch.Tensor
            Predicted values from the model.
        targets : torch.Tensor
            Target values.
        previous_values : torch.Tensor
            Previous values from the dataset.

        Returns
        -------
        torch.Tensor
            Loss value.
        """

        # Calculate MSE Loss
        mse = self.mse_loss(predictions, targets)

        # Calculate direction correctness
        prediction_directions = predictions - previous_values
        target_directions = targets - previous_values
        direction_correctness = (
            torch.sign(prediction_directions) == torch.sign(target_directions)
        ).float()

        # Calculate weighted direction loss
        direction_loss = self.weight * (1 - direction_correctness).mean()

        # Combine losses
        total_loss = mse + direction_loss

        return total_loss
