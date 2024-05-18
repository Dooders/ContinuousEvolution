import numpy as np
import torch
import torch.nn as nn


class Fitness:

    def __init__(self, criterion: nn.Module = nn.MSELoss()) -> None:
        self.criterion = criterion
        self.fitness_history = []

    def _evaluate(
        self, past_outputs: torch.Tensor, past_actual: torch.Tensor, y: torch.Tensor
    ) -> float:
        """
        Evaluate the fitness of a neural network on a dataset.

        Parameters
        ----------
        past_outputs : torch.Tensor
            Output prediction from the neural network.
        past_actual : torch.Tensor
            Actual target values.
        y : torch.Tensor
            Target values for the current input.

        Returns
        -------
        float
            Negative loss value as fitness. Lower loss is better.
        """
        past_actual = past_actual.unsqueeze(0)
        loss = self.criterion(past_outputs, past_actual, y)
        return -loss.item()  # Using negative loss as fitness, lower loss is better

    def evaluate(
        self, past_outputs: list, past_actual: torch.Tensor, y: torch.Tensor
    ) -> list:
        """
        Evaluate the fitness of each model in the population.

        Parameters
        ----------
        past_outputs : list
            List of past output predictions from the models.
        past_actual : torch.Tensor
            Actual target values.
        y : torch.Tensor
            Target values for the current input.

        Returns
        -------
        list
            List of fitness values for each model.
        """
        fitnesses = [
            self._evaluate(past_output, past_actual, y) for past_output in past_outputs
        ]
        self.fitness_history.append(fitnesses)
        return fitnesses

    def most_fit(self):
        #! Will need to update this to return the model with the highest fitness instead of the index
        return np.argmax(self.fitness_history[-1])
