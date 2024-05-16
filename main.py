import numpy as np
import torch

from crossover import AverageCrossover
from data import X, prices, y
from evolve import ContinuousEvolution
from lstm import LSTMModel
from mutate import GaussianMutation

current_input = X[1]
past_actual_target = y[0]
past_input = X[0]

# Parameters
max_cycles = 300
population_size = 100
num_parents = 40

settings = {"input_dim": 1, "hidden_dim": 3}
# settings = {}

# Start the evolutionary training
evo = ContinuousEvolution(
    model=LSTMModel,
    settings=settings,
    population=population_size,
    parents=num_parents,
    crossover_strategy=AverageCrossover(),
    mutation_strategy=GaussianMutation(),
)

past_outputs = []

if not past_outputs:
    past_outputs = [model(X[0]) for model in evo.population]
    past_actual_target = y[0]

fitnesses = [
    evo.evaluate_fitness(past_output, past_actual_target)
    for past_output in past_outputs
]
past_best_model = evo.best(
    past_input, past_actual_target
)  #! Get rid of this and just take off the fitness list which will be a sorted list by fitness
parents = evo.select_parents(evo.population, fitnesses, evo.parents)

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

plt.figure(figsize=(15, 5))
plt.plot(prices, label="Actual")
plt.plot(past_best_model(X).detach().numpy(), label="Predicted")
plt.legend()
plt.show()
