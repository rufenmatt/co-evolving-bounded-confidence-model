import node
import numpy as np
import pytest


def test_init():
    initial_opinion = 0.5
    neighbors = [2,3]
    C = 0.3
    node1 = node.Node(1, initial_opinion, C, neighbors=neighbors)
    node2 = node.Node(2, initial_opinion, C)
    assert node1.id == 1
    assert node1.initial_opinion == initial_opinion
    assert node1.current_opinion == initial_opinion
    assert node1.C == C
    assert node1.neighbors == neighbors
    assert len(node1.neighbors) == 2
    assert len(node2.neighbors) == 0
    assert node2.neighbors == []
    assert node2.C == C


@pytest.fixture
def example_node():
    id = 1
    initial_opinion = 0.5
    neighbors = [2,3]
    C = 0.3
    return node.Node(id,initial_opinion,C,neighbors)

def test_add_neighbor(example_node):
    example_node.add_neighbor(4)
    assert example_node.neighbors == [2,3,4]

def test_erase_neighbor(example_node):
    example_node.erase_neighbor(2)
    assert example_node.neighbors == [3]

def test_check_neighbor(example_node):
    assert example_node.check_if_neighbor(2) == True
    assert example_node.check_if_neighbor(4) == False

# Note: for this test function only the __compute_rewiring_prob method name
# is modified such that is it not a private method by removing one underscore.
# In the actual source code, it will be a private method.
def test_compute_rewiring_prob(example_node):
    X = [1,1,1,1,1]
    X[example_node.id] = example_node.current_opinion
    expected_rewiring_distr = [0.5, 0, 0, 0, 0.5]
    rewiring_distr = example_node._compute_rewiring_prob(X)
    assert np.all([rewiring_distr[i] == expected_rewiring_distr[i] for i in range(5)])

def test_rewire(example_node):
    X = [1,1,1,1,1]
    X[example_node.id] = example_node.current_opinion
    neighbors_before_rewiring = example_node.neighbors.copy()
    rng = np.random.default_rng()
    new_neighbor = example_node.rewire(X, rng)
    assert new_neighbor not in neighbors_before_rewiring
    assert ((new_neighbor==0 or new_neighbor==4))
    assert len(neighbors_before_rewiring) < len(example_node.neighbors)
