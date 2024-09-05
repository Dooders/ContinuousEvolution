import numpy as np
from sklearn.preprocessing import StandardScaler
import torch

from evolution.crossover import AverageCrossover
# from prices import prices
from evolution.evolve import ContinuousEvolution
from evolution.loss import DirectionalMSELoss
from models.lstm import LSTMModel
from evolution.mutation import GaussianMutation


import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from data.sine import sin_values

# repeat the sin_values twice
prices = np.concatenate([sin_values])

# Disabling gradient computation globally
torch.set_grad_enabled(False)

# Parameters
population_size = 100
num_parents = 2

settings = {"input_dim": 1, "hidden_dim": 3}

# Start the evolutionary training
evo = ContinuousEvolution(
    model=LSTMModel,
    criterion=torch.nn.MSELoss(),
    settings=settings,
    population_size=population_size,
    parent_count=num_parents,
    crossover_strategy=AverageCrossover(),
    mutation_strategy=GaussianMutation(),
)

results = []
for price in prices:
    value = torch.tensor(price, dtype=torch.float32)
    result = evo.run(value, 10) #! Add the window size as a parameter
    results.append(result[0].item())  #! Why are there two values in result?

# Plot the predictions from training data
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(prices, label="Actual")
plt.plot(results, label="Predictions")
plt.legend()
plt.show()


# Boxplot of the predictions
all_predictions = evo.output_buffer
plt.boxplot(all_predictions)
plt.show()