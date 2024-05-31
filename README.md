# ContinuousEvolution

A framework to facilitate continuous evolution of a population of agent models.  

## Required Components

### Fitness: Class

fitness: data object: list[float]

initial arguments: none
methods:
   @abstractmethod
   @classmethod
   evaluate(population, strategy) #class method
attributes: none?
properties: none? calcs? avg, median, etc

requirements:
- allow for user to select the specific fitness strategy. An extension of the base class. with abstract methods
- returns a list of flotes that represent a fitness score determended by the selected strategy. Highest means most fit
- does it need a fitness history????
- 

A method to evaluate the fitness of a population of agents, dependent on the goals established in the simulation.

The fitness is represented by a list of values that represent the "fitness" of the agent.

The user will select a Fitness strategy that is used for the simulation.

### Selection: Class

initial arguments:
methods:
attributes:
properties:

A method to select which agents continue into the next generation, or which agents will be used as parents for the next generation.

This class should return a list of agents.

The user will select the selection strategy to be used in the simulation.

### Crossover: Class

initial arguments:
methods:
attributes:
properties:

A method to combine the two parent parameters into a new agent.

Will return a new agent based on parent_x and parent_y.

The user will select the crossover strategy to use in the simulation.

### Mutation: Class

initial arguments:
methods:
attributes:
properties:

Method that applies a mutation to the parameters of an agent, done directly after crossover.

Returns a mutated agent.

The user will select a mutation strategy to use during the simulation.

### Agent Model: Class

initial arguments:
methods:
attributes:
properties:

The neural network architecture to use for the agent.

The user will select the model to use to represent the agent in the simulation.

### Simulation Engine: Class

initial arguments:
methods:
attributes:
properties:

The specific process to run for each cycle of the simulation.

It is the logic to evaluate fitness, select agents, produce new agents, mutate them, and make predictions.

Simulation engines are custom to the use case.  

* A simulation can create new agents every step or keep the selected in the simulation while removing unfit and replacing them.
* First testing with a cycle being one step, likely much better to have a cycle represent multiple steps to give each agent a fair shot at fitness.
