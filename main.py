import numpy as np
import torch

from crossover import AverageCrossover
from data import X, prices, y
from evolve import ContinuousEvolution
from loss import DirectionalMSELoss
from lstm import LSTMModel
from mutate import GaussianMutation

# Disabling gradient computation globally
torch.set_grad_enabled(False)

current_input = X[1]
past_actual_target = y[0]
past_input = X[0]

# Parameters
max_cycles = 300
population_size = 100
num_parents = 40

settings = {"input_dim": 1, "hidden_dim": 3}

# Start the evolutionary training
evo = ContinuousEvolution(
    model=LSTMModel,
    criterion=DirectionalMSELoss(),
    settings=settings,
    population_size=population_size,
    parent_count=num_parents,
    crossover_strategy=AverageCrossover(),
    mutation_strategy=GaussianMutation(),
)

past_outputs = []

if not past_outputs:
    past_outputs = [model(X[0]) for model in evo.population]
    past_actual_target = y[0]

fitnesses = evo.fitness.evaluate(past_outputs, past_actual_target, y[0])
best_model = evo.fitness.most_fit
parents = evo.select_parents(
    evo.population, fitnesses, evo.parent_count
)  #! Improve this process

next_generation = []
while len(next_generation) < evo.population_size:
    for i in range(len(parents)):
        for j in range(i + 1, len(parents)):
            child = evo.crossover(parents[i], parents[j])
            evo.mutate(child)
            next_generation.append(child)
            if len(next_generation) >= evo.population_size:
                break
evo.population = next_generation

print(
    f"Average fitness: {np.mean(fitnesses):.4f} Best: {max(fitnesses):.4f} Worst: {min(fitnesses):.4f}"
)

with torch.no_grad():
    new_outputs = [model(current_input) for model in evo.population]
    past_outputs = new_outputs

# Plot the predictions from training data
import matplotlib.pyplot as plt

predictions = evo.population[best_model](X).detach().numpy()
plt.figure(figsize=(12, 6))
plt.plot(prices, label="Actual")
plt.plot(predictions, label="Predictions")
plt.legend()
plt.show()
