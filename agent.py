from utils import seed


class Agent:
    """
    Base class for all agents in the simulation.
    """

    def __init__(self) -> None:
        self.id = seed.id()
