import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import torch
from tqdm import tqdm

from crossover import Average
from evolve import ContinuousEvolution
from loss import DirectionalMSELoss
from models.simple import SimpleSequentialNetwork
from mutation import Gaussian
from price_data import price_df
from selection import Rank
from utils import set_seed

# Disabling gradient computation globally
torch.set_grad_enabled(False)

params = {
    "population_size": 100,
    "num_parents": 2,
    "input_size": 9,
    "seed": "seed_123",
    "experiment": "price-sim",
    "run_name": "run_1",
    "description": "First run",
    "step_size": 1,
}

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(params["experiment"])


# Start MLflow run
with mlflow.start_run(run_name=params["run_name"], description=params["description"]):

    # Log parameters
    mlflow.log_params(params)

    set_seed(params["seed"])

    # Start the evolutionary training
    evo = ContinuousEvolution(
        agent_type=SimpleSequentialNetwork,
        fitness=DirectionalMSELoss(),
        settings={"input_size": params["input_size"]},
        population_size=params["population_size"],
        parent_count=params["num_parents"],
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
    
    step = 0

    # Go through each row of df_scaled
    for row in tqdm(price_df.iterrows(), total=price_df.shape[0]):
        X = list(row[1])
        evo.input_buffer.append(X[0])
        X = torch.tensor(X).float()

        if len(evo.input_buffer) >= 2:
            price_went_down = evo.input_buffer[-1] < evo.input_buffer[-2]
            fitness = evaluate_agents(evo.output_buffer[-1], price_went_down)

            # Store and log fitness metrics
            avg_fitness = np.mean(fitness)
            best_fitness = max(fitness)
            worst_fitness = min(fitness)

            avg_fitness_history.append(avg_fitness)
            best_fitness_history.append(best_fitness)
            worst_fitness_history.append(worst_fitness)

            mlflow.log_metric("avg_fitness", avg_fitness)
            mlflow.log_metric("best_fitness", best_fitness)
            mlflow.log_metric("worst_fitness", worst_fitness)

            # Reproduce population
            next_generation = []

            parents = Rank.select(
                population=list(zip(evo.population, fitness)), n=evo.parent_count
            )

            for _ in range(evo.population_size):
                parent_x, parent_y = np.random.choice(parents, 2, replace=False)
                child = evo.crossover_strategy.crossover(parent_x, parent_y)
                evo.mutate(child)
                next_generation.append(child)

            evo.population = next_generation

        with torch.no_grad():
            new_outputs = [model(X).item() for model in evo.population]
            evo.output_buffer.append(new_outputs)

        step += 1

    # After the main loop, create and log the chart
    fig = plt.figure(figsize=(20, 12))

    # Create a gridspec for more control over subplot placement
    gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[3, 1], hspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])

    # Calculate overall average fitness
    overall_avg_fitness = np.mean(avg_fitness_history)

    # Plot fitness metrics
    ax1.plot(avg_fitness_history, label="Average Fitness")
    # ax1.plot(best_fitness_history, label="Best Fitness")
    # ax1.plot(worst_fitness_history, label="Worst Fitness")
    ax1.axhline(
        y=overall_avg_fitness, color="r", linestyle="--", label="Overall Average"
    )
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

    # Add parameters to the right of the plot
    param_text = "\n".join([f"{k}: {v}" for k, v in params.items()])
    param_ax = fig.add_subplot(gs[:, 1])
    param_ax.axis("off")
    param_ax.text(
        0, 0.5, f"Parameters:\n\n{param_text}", va="center", ha="left", wrap=True
    )

    plt.tight_layout()

    # Adjust the layout to prevent overlap
    fig.subplots_adjust(hspace=0.3)

    # Save and log the chart as an artifact
    chart_path = "fitness_chart.png"
    plt.savefig(chart_path, bbox_inches="tight")
    mlflow.log_artifact(chart_path)

    # Optionally log the final model or other outputs
    # mlflow.pytorch.log_model(evo, "evolution_model")

    # Show the plot
    plt.show()
