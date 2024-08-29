import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import mlflow
import mlflow.pytorch

from crossover import Average
from evolve import ContinuousEvolution
from loss import DirectionalMSELoss
from models.simple import SimpleSequentialNetwork
from mutation import Gaussian
from price_data import price_df
from selection import Rank

# Disabling gradient computation globally
torch.set_grad_enabled(False)

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("price-sim")

# Parameters
population_size = 20
num_parents = 4

settings = {"input_size": 9}

# Start MLflow run
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("population_size", population_size)
    mlflow.log_param("num_parents", num_parents)
    mlflow.log_param("input_size", settings["input_size"])

    # Start the evolutionary training
    evo = ContinuousEvolution(
        agent_type=SimpleSequentialNetwork,
        fitness=DirectionalMSELoss(),
        settings=settings,
        population_size=population_size,
        parent_count=num_parents,
        crossover_strategy=Average(),
        mutation_strategy=Gaussian(),
    )

    def evaluate_agents(outputs, price_went_down):
        scores = []
        for output in outputs:
            if price_went_down:
                score = output
            else:
                score = 1 - output
            scores.append(score)
        return scores

    # Variables for tracking fitness history
    avg_fitness_history = []
    best_fitness_history = []
    worst_fitness_history = []

    # Go through each row of df_scaled
    for row in tqdm(price_df.iterrows(), total=price_df.shape[0]):

        X = list(row[1])
        evo.input_buffer.append(X[0])
        X = torch.tensor(X).float()

        if len(evo.input_buffer) >= 2:
            price_went_down = evo.input_buffer[-1] < evo.input_buffer[-2]
            fitness = evaluate_agents(evo.output_buffer[-1], price_went_down)
            avg_fitness = np.mean(fitness)
            best_fitness = max(fitness)
            worst_fitness = min(fitness)

            # Store fitness metrics
            avg_fitness_history.append(avg_fitness)
            best_fitness_history.append(best_fitness)
            worst_fitness_history.append(worst_fitness)

            # Log fitness metrics to MLflow
            mlflow.log_metric("avg_fitness", avg_fitness)
            mlflow.log_metric("best_fitness", best_fitness)
            mlflow.log_metric("worst_fitness", worst_fitness)

            next_generation = []

            parents = Rank.select(
                population=list(zip(evo.population, fitness)), n=evo.parent_count
            )

            for i in range(evo.population_size):
                parent_x, parent_y = np.random.choice(parents, 2, replace=False)
                child = evo.crossover_strategy.crossover(parent_x, parent_y)
                evo.mutate(child)
                next_generation.append(child)

            evo.population = next_generation

        with torch.no_grad():
            new_outputs = [model(X).item() for model in evo.population]
            evo.output_buffer.append(new_outputs)

    # After the main loop, create and log the chart
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]}
    )

    # Plot fitness metrics
    ax1.plot(avg_fitness_history, label="Average Fitness")
    # ax1.plot(best_fitness_history, label="Best Fitness")
    # ax1.plot(worst_fitness_history, label="Worst Fitness")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness")
    ax1.set_title("Fitness Metrics Over Time")
    ax1.legend()
    ax1.grid(True)

    # Plot actual prices
    ax2.plot(price_df.index, price_df.iloc[:, 0], label="Actual Price", color="green")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Price")
    ax2.set_title("Actual Prices")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    # Save and log the chart as an artifact
    chart_path = "fitness_chart.png"
    plt.savefig(chart_path)
    mlflow.log_artifact(chart_path)

    # Optionally log the final model or other outputs
    # mlflow.pytorch.log_model(evo, "evolution_model")

    # Show the plot
    plt.show()
