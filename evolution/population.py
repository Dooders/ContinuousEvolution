import networkx as nx

from evolution.agent import Agent

AgentID = str
FitnessScore = float

FitnessList = list[tuple["AgentID", "FitnessScore"]]


class Population(list):
    """
    List of Agents sorted desc by fitness.

    List[Agent]
    """

    def __init__(self, *args):
        super().__init__(*args)

    def sort_by_fitness(self):
        """
        Sort the agents by fitness in descending order.
        """
        return sorted(self, key=lambda x: x.fitness, reverse=True)

    def sorted(self) -> "FitnessList":
        """
        Returns
        -------
        "FitnessList"
            list[tuple["AgentID", "FitnessScore"]]
            Sorted by descending order.
        """
        return FitnessList(
            [(agent.id, agent.fitness) for agent in self.sort_by_fitness()]
        )


class CollectiveState: 
    """
    A directed graph of agents and their relationships. 
    
    With a temporal component with the ability to save state snapshots
    
    Family is currently the only relationship type. Serving as a family tree.
    Maybe called thread instead of edge
    
    #! maybe wrongly named, maybe this should be the CollectiveState class
    #! and it basically handles the state of the agents and their relationships
    """
    def __init__(self):
        self.graph = nx.DiGraph()
        self.active_states = [] #! this will be the state_buffer, which is a deque

    def add_agent(self, agent):
        self.graph.add_node(agent.id, agent=agent)

    def add_relationship(self, agent1_id, agent2_id, relationship_type):
        self.graph.add_edge(agent1_id, agent2_id, relationship=relationship_type)
        self.graph.add_edge(agent2_id, agent1_id, relationship=relationship_type)

    def save_state(self, agent_id, state):
        agent = self.graph.nodes[agent_id]["agent"]
        agent.add_state(state)
        self.active_states.append((agent_id, state))

    def progress_time(self, current_time):
        # Implement logic to move states from memory to persistent storage
        pass

#!!!!!!! This below is being done above with CollectiveState
class Collection:  # Removed the inheritance from "TemporalDirectedGraph"
    """
    List of Agents sorted desc by fitness.
    """
    #! methods : fill and replace, and create
    #! reproduce sends the instructions, population makes the agents, replacing or filling the population to its max size
    #! i mean that is population control, right? Manages and influences the population of agents

    def __init__(self, *args):
        #! How do I make it sorted automatically?
        super().__init__(*args)

    def fitness(self) -> FitnessList:
        """
        Return the fitness of the agents in the collection.
        """
        return [agent.fitness for agent in self]


#! its a framework to have maximum mechanisms to setup, run, and observing behavior of agents
#! invite people to get involved in the project

#! This will have the output_buffer (Actions?)

#! Need to eventually make this real time and not sequential time. The agent is put on the board, the motor was wound, and now you see how it will do in the Environment over Time

#! the db is the hard storage, the graph is the in memory storage, the agent is the object that is being observed
#! The data object is Collection (or collective :) of agents in a graph like db and designed to be used in a temporal manner

