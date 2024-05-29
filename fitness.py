import numpy as np
import torch
import torch.nn as nn


class Fitness:
    """
    Class to evaluate the fitness of a neural network on a dataset.

    Parameters
    ----------
    criterion : nn.Module, optional
        Loss function to use, by default nn.MSELoss.

    Attributes
    ----------
    criterion : nn.Module
        Loss function to use.
    fitness_history : list
        List of fitness values at each cycle.

    Methods
    -------
    evaluate(past_outputs: list, past_actual: torch.Tensor, y: torch.Tensor) -> list:
        Evaluate the fitness of each model in the population.
    most_fit:
        Return the index of the model with the highest fitness.
    """

    def __init__(self, criterion: nn.Module = nn.MSELoss()) -> None:
        self.criterion = criterion
        self.fitness_history = []

    def _evaluate(self, past_outputs: torch.Tensor, past_actual: torch.Tensor) -> float:
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
        loss = self.criterion(past_outputs, past_actual)
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
            self._evaluate(past_output, past_actual) for past_output in past_outputs
        ]
        self.fitness_history.append(fitnesses)
        return fitnesses

    @property
    def most_fit(self):
        #! Will need to update this to return the model with the highest fitness instead of the index
        return np.argmax(self.fitness_history[-1])


class TradingFitness(Fitness):
    """
    Will return if the agent should be removed or not based on the fitness value.

    Need to figure out how to score the agent based on the outcome.

    Fitness scenarios:
    - Agent sells, loses money (removed)
    - Agent sells, makes money (remains)
    - Agent does not sell, misses out on  R profit (removed)
    - Agent does not sell, avoids loss (remains)
    - Agent sells, has loss but has a good history (remains?)
    - Agent has N streak of losses, even with net profit (removed)

    N is the number of losses in a row before the agent is removed.
    R is the profit that the agent missed out on.

    So the class will need the agent's history and the results from the agent's actions.
    """

    def __init__(self, criterion: nn.Module = nn.MSELoss()) -> None:
        super().__init__(criterion)

        self.agent_history = []

    def evaluate(self, past_outputs: list, price_went_down: bool) -> list:
        """
        Evaluates the agents based on their predictions and the actual price movement.

        If the price went down, the agents are scored based on how close their
        predictions were to 1.

        If the price went up, the agents are scored based on how close their
        predictions were to 0.

        Parameters
        ----------
        outputs : list of float
            The outputs of the agents' predictions.
        price_went_down : bool
            Indicates if the price went down (True) or up (False).

        Returns
        -------
        scores : list of int
            The evaluation scores ranging from 0 to 99.
        """
        scores = []
        for output in past_outputs:
            if price_went_down:
                # Higher scores for outputs closer to 1
                score = output
            else:
                # Higher scores for outputs closer to 0
                score = 1 - output
            scores.append(score)

        self.fitness_history.append(scores)

        return scores

    @property
    def most_fit(self) -> int:
        """
        Returns
        -------
        int
            The index of the agent with the highest fitness.
        """
        return np.argmax(self.fitness_history[-1])
