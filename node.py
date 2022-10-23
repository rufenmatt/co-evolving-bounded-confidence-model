import numpy as np

class Node:
    # For constant C
    #def __init__(self, id, initial_opinion, neighbors=None):
    # For uniform C
    def __init__(self, id, initial_opinion, C, neighbors=None):
        self.id = id
        self.neighbors = neighbors if neighbors is not None else []
        self.initial_opinion = initial_opinion
        self.current_opinion = initial_opinion
        self.C = C
        self.total_opinion_change = 0

    def add_neighbor(self, id):
        self.neighbors.append(id)

    def erase_neighbor(self, id):
        self.neighbors.remove(id)

    def check_if_neighbor(self, id):
        return id in self.neighbors

    def update_opinion(self, new_x):
        self.total_opinion_change += abs(self.current_opinion - new_x)
        self.current_opinion = new_x

    def rewire(self, X, rng):
        # Compute rewiring probabilities
        rewiring_prob = self._compute_rewiring_prob(X)
        # Select one from random
        new_neighbor = rng.choice(range(len(rewiring_prob)), p=rewiring_prob)
        # Add new neighbor to list of neighbors
        self.add_neighbor(new_neighbor)
        # Return the new neighbor
        return new_neighbor


    def _compute_rewiring_prob(self, X):
        # Compute distances using the L2 metric
        distances = np.array([(self.current_opinion - x)**2 for x in X])

        # Compute the rewiring probability distribution
        rewiring_distr = 1 - distances
        rewiring_distr[self.id] = 0 # Set probability for x->x to 0 to prevent self-edges
        rewiring_distr[self.neighbors] = 0 # Avoid duplicate edges and length-2 cycles
        # Normalize the distribution
        A = np.sum(rewiring_distr)
        rewiring_distr = (1/A) * rewiring_distr

        return rewiring_distr
