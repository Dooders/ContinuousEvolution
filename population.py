class Population(list):  #! will be a heap in the future
    """
    A list of agents representing the population.

    Sorted desc by fitness.
    """

    def __init__(self, *args):
        super().__init__(*args)
