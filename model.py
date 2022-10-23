import numpy as np
from numpy.random import default_rng, SeedSequence, RandomState, MT19937
import random
import networkx as nx
from networkx.generators.random_graphs import fast_gnp_random_graph
import _pickle as cPickle
import bz2
import csv

from node import Node

class Model:
    def __init__(self, seedseq, **kwparams):
        # Set random state to ensure reproducability
        self.seedseq = seedseq      # Set the SeedSequence
        try:
            self.spawn_key = seedseq.spawn_key[0]
        except:
            self.spawn_key = None
        # Each instance gets its own RNG for stochasic elements
        self.rng = default_rng(seedseq)
        # Set state of the random module
        random.seed(seedseq)
        self.random_state = RandomState(MT19937(seedseq))

        # Set model parameters using kwparams
        self.maxsteps = kwparams['maxsteps']    # Bail-out time
        self.tolerance = kwparams['tolerance']  # Convergence tolerance
        self.alpha = kwparams['alpha']      # Convergence speed parameter
        self.C = kwparams['C']              # Confidence bound
        self.tolerance_upper = kwparams['tolerance_upper']   # Upper confidence bound
        self.tolerance_lower = kwparams['tolerance_lower']   # Lower confidence bound
        self.beta = kwparams['beta']        # Opinion tolerance threshold
        self.synthetic = kwparams['synthetic']  # Use real-world or synthetic network data
        self.dataset = kwparams['dataset']  # Dataset
        self.N = kwparams['N']              # Number of nodes for synthetic networks
        self.p = kwparams['p']              # P in G(N,p)
        self.M = kwparams['M']              # Number of edges to rewire at t
        self.K = kwparams['K']              # Number of node pairs to adjust opinions at t
        self.trial = kwparams['trial']      # Trial identifer
        self.fulltimeseries = kwparams['fulltimeseries']        # Timeseries logging

        # TO DO: Add more elegant way of switching between real-world and synthetic networks
        if self.synthetic == 'no':
            # Load network and opinion data
            self.__load_data()

            # Set N for the respective dataset
            if self.dataset == 'Reddit':
                n_nodes = 556

            if self.dataset == 'Twitter':
                n_nodes = 548

            self.N = n_nodes
            self.N = 10   # N for test run
        else:
            # Generate an initial setup and set attributes
            self.__initialize_system()

        # Initialize additional attributes for data storage
        if self.fulltimeseries:
            self.X_data = np.ndarray((self.maxsteps,self.N))    # Opinion time series
            self.X_data[0,:] = self.X           # Initial opinion profiles
            self.edge_changes = []              # Edge changes
        else:
            self.X_data = np.ndarray((int(self.maxsteps/500)+1,self.N))    # Opinions at every 500th step
            self.X_data[0,:] = self.X           # Initial opinion profiles
            self.G_snapshots = []        # Network snapshots at every 500th step
        self.num_discordant_edges = np.empty(self.maxsteps)      # Discordant edges
        self.stationary_counter = 0         # Stationary state
        self.stationary_marker = 0          # Stationary state logging
        self.convergence_time = None        # Convergence time


        # If beta=1.0, there is no rewiring step in the proceess
        self.rewiring_enabled = False if int(kwparams['beta']==1) else True



    def run(self):
        t=0

        #======== INNER HELPER FUNCTIONS TO RUN THE MODEL ========
        def rewire_step():
            if self.rewiring_enabled:

                # Compute discordant edges
                discordant_edges = [(i,j) for (i,j) in self.edges if abs(self.X[i] - self.X[j]) > self.beta]
                self.num_discordant_edges[t] = len(discordant_edges)


                # If len(discordant_edges) >= M, choose M at random using self.rng
                # else, choose all of the discordant edges to rewire
                if len(discordant_edges) > self.M:
                    idx = self.rng.choice(a=len(discordant_edges), size=self.M, replace=False)
                    edges_to_dissolve = [discordant_edges[i] for i in idx]
                else:
                    edges_to_dissolve = discordant_edges


                # Dissolve and rewire edges
                for edge in edges_to_dissolve:
                    # Dissolve edge
                    self.edges.remove(edge)
                    i = edge[0]; j=edge[1]
                    self.nodes[i].erase_neighbor(j)
                    self.nodes[j].erase_neighbor(i)

                    # Select i or j to rewire
                    random_node_selector = self.rng.integers(2)
                    i = i if random_node_selector==0 else j
                    selected_node = self.nodes[i]
                    new_neighbor = selected_node.rewire(self.X, self.rng)
                    self.nodes[new_neighbor].add_neighbor(i)
                    new_edge = (i, new_neighbor)

                    # Store data
                    self.edges.append(new_edge)
                    if self.fulltimeseries:
                        self.edge_changes.append((t, edge, new_edge))

                # Future simulations should be made with shorter steps (e.g, 250)
                if (self.fulltimeseries==False) and (t in [500,1000,2000,5000,7500,10000,15000,20000]):
                    G = nx.Graph()
                    G.add_nodes_from(range(self.N))
                    G.add_edges_from(self.edges)
                    self.G_snapshots.append((t, G))



        def dw_step():
            # Select K node pairs at random using self.rng
            idx = self.rng.integers(low=0, high=len(self.edges), size=self.K)
            nodepairs = [self.edges[i] for i in idx]

            # For each pair, update opinions both at the model level and node level
            X_new = self.X.copy()
            for u,w in nodepairs:
                # Updating rule for uniformly distributed C
                if abs(self.X[u] - self.X[w]) <= min(self.nodes[u].C, self.nodes[w].C):
                    X_new[u] = self.X[u] + self.alpha*(self.X[w] - self.X[u])
                    X_new[w] = self.X[w] + self.alpha*(self.X[u] - self.X[w])
                    self.nodes[u].update_opinion(X_new[u])
                    self.nodes[w].update_opinion(X_new[w])
                if abs(self.X[u] - self.X[w]) <= self.nodes[u].C:
                    X_new[u] = self.X[u] + self.alpha*(self.X[w] - self.X[u])
                    self.nodes[u].update_opinion(X_new[u])
                if abs(self.X[u] - self.X[w]) <= self.nodes[w].C:
                    X_new[w] = self.X[w] + self.alpha*(self.X[u] - self.X[w])
                    self.nodes[w].update_opinion(X_new[w])

                # Updating rule for constant C
                # if abs(self.X[u] - self.X[w]) <= self.C:
                #     X_new[u] = self.X[u] + self.alpha*(self.X[w] - self.X[u])
                #     X_new[w] = self.X[w] + self.alpha*(self.X[u] - self.X[w])
                # self.nodes[u].update_opinion(X_new[u])
                # self.nodes[w].update_opinion(X_new[w])

            # Update data storage
            self.X_prev = self.X.copy()
            self.X = X_new
            if self.fulltimeseries:
                self.X_data[t+1,:] = X_new
            elif (t%500==0):
                t_prime = int(t/500)
                self.X_data[t_prime+1] = X_new

        def check_convergence():
            state_change = np.sum(np.abs(self.X - self.X_prev))
            self.stationary_counter = self.stationary_counter+1 if state_change < self.tolerance else 0
            self.stationary_marker = 1 if self.stationary_counter >= 100 else 0


        #==================== ACTUALLY RUN THE MODEL ====================

        while ((t < self.maxsteps-1) & (self.stationary_marker != 1)):
            if self.rewiring_enabled:
                rewire_step()
            dw_step()
            check_convergence()
            t+=1

        self.convergence_time = t
        self.save_model()

    # ===================================================
    # AREA FOR PRIVATE HELPER FUNCTIONS BELOW
    # ===================================================


    def __load_data(self):
        """
        This helper function is for loading the initial network and opinions
        from Reddit or Twitter.

        Sets the following attributes:
        - self.X : an array of length N
        - self.initial_edges : a list of tuples of edges
        - self.nodes : a list of length N containing Node objects
        - self.edges : a list of tuples of edges
        """

        # draw initial opinions from Unif[0,1] using the rng of the caller object
        # X = self.rng.random(self.N)
        # generate a G(N,p) random graph using the random state of the caller object
        # G = nx.fast_gnp_random_graph(n=self.N, p=self.p, seed=self.random_state, directed=False)

        A = np.zeros([self.N, self.N])
        X = {i:[] for i in range(self.N)}

        # with open("data/{self.dataset}/edges_{self.dataset}.txt", "r") as f:
        #     reader = csv.reader(f, delimiter="\t")
        #     for u, v in reader:
        #         A[int(u)-1, int(v)-1] += 1
        #         A[int(v)-1, int(u)-1] += 1

        # with open("data/{self.dataset}/{self.dataset}_opinion.txt","r") as f:
        #     reader = csv.reader(f,delimiter="\t")
        #     for u,v,w in reader:
        #         X[int(u)-1].append(float(w))

        # with open("data/reddit/edges_reddit.txt", "r") as f:
        #     reader = csv.reader(f, delimiter="\t")
        #     for u, v in reader:
        #         A[int(u)-1, int(v)-1] += 1
        #         A[int(v)-1, int(u)-1] += 1

        # with open("data/reddit/reddit_opinion.txt","r") as f:
        #     reader = csv.reader(f,delimiter="\t")
        #     for u,v,w in reader:
        #         X[int(u)-1].append(float(w))

        with open("data/test/edges_test.txt", "r") as f:
            reader = csv.reader(f, delimiter="\t")
            for u, v in reader:
                A[int(u)-1, int(v)-1] += 1
                A[int(v)-1, int(u)-1] += 1

        with open("data/test/test_opinion.txt","r") as f:
            reader = csv.reader(f,delimiter="\t")
            for u,v,w in reader:
                X[int(u)-1].append(float(w))

        X = [np.mean(X[i]) for i in range(self.N)]
        X = np.array(X)

        # Transform adjacency matrix representation into a graph
        G = nx.from_numpy_array(A)

        # Generate a uniform distribution of N confidence bounds
        C_set = self.rng.uniform(self.tolerance_lower, self.tolerance_upper, self.N).tolist()

        # Initialize Node objects and store these in an array
        nodes = []
        for i in range(self.N):
            n_neigh = list(G[i])
            # assign a confidence bound to a node at random and remove from list
            C = self.rng.choice(C_set)
            C_set.pop(C_set.index(C))
            n = Node(id=i, initial_opinion=X[i], C=C, neighbors=n_neigh)
            nodes.append(n)

        # Get list of edges
        edges = [(u,v) for u,v in G.edges()]

        # Set these values as attributes of self
        # self.X : an array of length N
        # self.initial_edges : a list of tuples of edges
        # self.nodes : a list of length N containing Node objects
        # self.edges : a list of tuples of edges
        self.X = X
        self.initial_edges = edges.copy()
        self.nodes = nodes
        self.edges = edges.copy()

    def __initialize_system(self):
        """
        This helper function is for generating an initial system of N individuals
        with uniformly-distributed opinion drawn using self.rng and random connections
        among individuals are created using NetworkX and self.random_state.

        Sets the following attributes:
        - self.X : an array of length N
        - self.initial_edges : a list of tuples of edges
        - self.nodes : a list of length N containing Node objects
        - self.edges : a list of tuples of edges
        """

        # Draw initial opinions from Unif[0,1] using the rng of the caller object
        X = self.rng.random(self.N)

        # Generate a G(N,p) random graph using the random state of the caller object
        G = nx.fast_gnp_random_graph(n=self.N, p=self.p, seed=self.random_state, directed=False)

        # Generate a uniform distribution of N confidence bounds
        C_set = self.rng.uniform(self.tolerance_lower, self.tolerance_upper, self.N).tolist()

        # Initialize Node objects and store these in an array
        nodes = []
        for i in range(self.N):
            n_neigh = list(G[i])
            # Assign a confidence bound to a node at random and remove from list
            C = self.rng.choice(C_set)
            C_set.pop(C_set.index(C))
            n = Node(id=i, initial_opinion=X[i], C=C, neighbors=n_neigh)
            # For constant C
            #n = Node(id=i, initial_opinion=X[i], neighbors=n_neigh)
            nodes.append(n)

        # Get list of edges
        edges = [(u,v) for u,v in G.edges()]

        # Set these values as attributes of self
        self.X = X
        self.initial_X = X
        self.initial_edges = edges.copy()
        self.nodes = nodes
        self.edges = edges.copy()

    # ===================================================
    # PUBLIC HELPER FUNCTIONS BELOW
    # ===================================================


    def get_edges(self, t=None):
        """Returns a list of edges in the system's network at time t.
		For example, if t==1, then this returns the edges after the first round of rewiring has been executed.
		If t is None or is greater than the convergence time, the method returns the most recent snapshot.
		This method should only be used after the method run() has been called already.
		If the model was set to not save timeseries data, this will return the most recent snapshot.

		Args:
			t (int, optional): Timestep at which to get the network snapshot. Defaults to None.

		Returns:
		   a list of edges in the system at time t.
		"""
        """
		Note:
		I found a bug where self.timeseries should be self.fulltimeseries (now fixed)
		although this was discovered after simulations have been run, so this function
		is pretty much useless unless (t==None) or (t >= self.convergence_time) for the
		simulation data that I have (new simulation runs won't be affected)

		For the old simulation data, just create a function like this outside of the class
		as a work around.
		"""

		# When t==None, return the current snapshot of the network

        if (t==None) or (t >= self.convergence_time) or (self.timeseries==False):
            return self.edges.copy()
		# Elife t==0, return the original network
        elif t==0:
            return self.initial_edges.copy()
		# Else, construct a snapshot of the network at time t using the recorded edge changes
        else:
			# Find all the edge changes up until the highest T where T < t
            edges = self.initial_edges.copy()
            edge_changes = [(T,e1,e2) for (T,e1,e2) in self.edge_changes if T<t]
			# Iteratively make changes to the network, starting from initial edges
            for (T,old_edge,new_edge) in edge_changes:
                edges.remove(old_edge)
                edges.append(new_edge)
            return edges

    def get_network(self, t=None):
        """Returns a NetworkX Graph object that is a snapshot of the system's network at time t.
        For example, if t==1, then this returns the network after the first round of rewiring has been executed.
        If t is None or is greater than the convergence time, the method returns the most recent snapshot.
        This method should only be used after the method run() has been called already.
        If the model was set to not save timeseries data, this will return the most recent snapshot.

        Args:
            t (int, optional): Timestep at which to get the network snapshot. Defaults to None.

        Returns:
            NetworkX Graph object: A graph with N nodes and edges in the system at time t.
        """

        G = nx.Graph()
        G.add_nodes_from(range(self.N))
        edges = self.get_edges(t)
        G.add_edges_from(edges)
        return G


    def save_model(self):
        """
        Saves the model in a compressed file format (BZ2) using cPickle.
        """

        # Save only the rows of X_data that have been filled (i.e., remove 0-rows)
		#self.X_data = self.X_data[~np.all(self.X_data == 0, axis=1)]
        if self.fulltimeseries:
            self.X_data = self.X_data[:self.convergence_time, :]
        else:
            self.X_data = self.X_data[:int(self.convergence_time/500)+1, :]
        self.num_discordant_edges = self.num_discordant_edges[:self.convergence_time-1]
        self.num_discordant_edges = np.trim_zeros(self.num_discordant_edges)

        C = f"{self.C:.2f}".replace('.','')
        tolerance_lower = f"{self.tolerance_lower:.2f}".replace('.','')
        tolerance_upper = f"{self.tolerance_upper:.2f}".replace('.','')
        beta = f"{self.beta:.2f}".replace('.','')
        # For uniformly distributed C
        filename = f"data-50-trials/C_lower{tolerance_lower}_C_upper{tolerance_upper}_beta_{beta}_trial_{self.trial}_spk_{self.spawn_key}.pbz2"
        # For constant C
        #filename = f"data-50-trials/C_{C}_beta_{beta}_trial_{self.trial}_spk_{self.spawn_key}.pbz2"
        with bz2.BZ2File(filename, 'w') as f:
            cPickle.dump(self, f)
