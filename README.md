# ContinuousEvolution

A framework to facilitate continuous evolution of a population of agent models.  

## Required Components

### Fitness: Class

Intended to be used as a base class to be inherited where the evaluate method returns a Population object of the agent updated fitness scores

A Fitness subclass is passed to the ContinuousEvolution object

The method must evaluate the fitness of the agent and set its fitness to it's score, this is used to sort the list

#### methods

```python
@classmethod
@abstractmethod
evaluate(population: 'Population') -> 'Population' #sorted by fitness
   """ 
   'Population' is a list of 'Agents' sorted desc by fitness 
   """
```

---

### Selection: Class

initial arguments:
methods:
attributes:
properties:

A method to select which agents continue into the next generation, or which agents will be used as parents for the next generation.

This class should return a list of agents.

The user will select the selection strategy to be used in the simulation.

---

### Crossover: Class

initial arguments:
methods:
attributes:
properties:

A method to combine the two parent parameters into a new agent.

Will return a new agent based on parent_x and parent_y.

The user will select the crossover strategy to use in the simulation.

---

### Mutation: Class

initial arguments:
methods:
attributes:
properties:

Method that applies a mutation to the parameters of an agent, done directly after crossover.

Returns a mutated agent.

The user will select a mutation strategy to use during the simulation.

---

### Agent: Class

initial arguments:
methods:
attributes: fitness, parents, children, id, ???
properties:

The neural network architecture to use for the agent.

The user will select the model to use to represent the agent in the simulation.

---

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

---

### Population class

Heap?????
Holds the population
holds token_pool

#### methods

   avg(self)
   median(self)
   max(self)
   min(self)
   top(self, n: int)

### Phases

Trade Buy
Fitness
Selection
Recombination
Mutation
Trade Decision
Trade Sell
