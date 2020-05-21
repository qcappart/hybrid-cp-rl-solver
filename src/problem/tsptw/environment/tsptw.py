import networkx as nx
import random
import heapq
import numpy as np
import torch

class TSPTW:

    def __init__(self, n_city, travel_time, x_coord, y_coord, time_windows):
        """
        Create an instance of the TSPTW problem
        :param n_city: number of cities
        :param travel_time: travel time matrix between the cities
        :param x_coord: list of x-pos of the cities
        :param y_coord: list of y-pos of the cities
        :param time_windows: list of time windows of all cities ([lb, ub] for each city)
        """

        self.n_city = n_city
        self.travel_time = travel_time
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.time_windows = time_windows
        self.graph = self.build_graph()

    def build_graph(self):
        """
        Build a networkX graph representing the TSPTW instance. Features on the edges are the distances
        and 4 binary values stating if the edge is part of the (1, 5, 10, 20) nearest neighbors of a node?
        :return: the graph
        """

        g = nx.DiGraph()

        for i in range(self.n_city):

            cur_travel_time = self.travel_time[i][:]

            # +1 because we remove the self-edge (cost 0)
            k_min_idx_1 = heapq.nsmallest(1 + 1, range(len(cur_travel_time)), cur_travel_time.__getitem__)
            k_min_idx_5 = heapq.nsmallest(5 + 1, range(len(cur_travel_time)), cur_travel_time.__getitem__)
            k_min_idx_10 = heapq.nsmallest(10 + 1, range(len(cur_travel_time)), cur_travel_time.__getitem__)
            k_min_idx_20 = heapq.nsmallest(20 + 1, range(len(cur_travel_time)), cur_travel_time.__getitem__)

            for j in range(self.n_city):

                if i != j:

                    is_k_neigh_1 = 1 if j in k_min_idx_1 else 0
                    is_k_neigh_5 = 1 if j in k_min_idx_5 else 0
                    is_k_neigh_10 = 1 if j in k_min_idx_10 else 0
                    is_k_neigh_20 = 1 if j in k_min_idx_20 else 0

                    weight = self.travel_time[i][j]
                    g.add_edge(i, j, weight=weight, is_k_neigh_1=is_k_neigh_1, is_k_neigh_5=is_k_neigh_5,
                               is_k_neigh_10=is_k_neigh_10, is_k_neigh_20=is_k_neigh_20)

        assert g.number_of_nodes() == self.n_city

        return g

    def get_edge_feat_tensor(self, max_dist):
        """
        Return a tensor of the edges features.
        As the features for the edges are not state-dependent, we can pre-compute them
        :param max_dist: the maximum_distance possible given the grid-size
        :return: a torch tensor of the features
        """

        edge_feat = [[e[2]["weight"] / max_dist,
                      e[2]["is_k_neigh_1"],
                      e[2]["is_k_neigh_5"],
                      e[2]["is_k_neigh_10"],
                      e[2]["is_k_neigh_20"]]
                     for e in self.graph.edges(data=True)]

        edge_feat_tensor = torch.FloatTensor(edge_feat)

        return edge_feat_tensor


    @staticmethod
    def generate_random_instance(n_city, grid_size, max_tw_gap, max_tw_size,
                                 is_integer_instance, seed):
        """
        :param n_city: number of cities
        :param grid_size: x-pos/y-pos of cities will be in the range [0, grid_size]
        :param max_tw_gap: maximum time windows gap allowed between the cities constituing the feasible tour
        :param max_tw_size: time windows of cities will be in the range [0, max_tw_size]
        :param is_integer_instance: True if we want the distances and time widows to have integer values
        :param seed: seed used for generating the instance. -1 means no seed (instance is random)
        :return: a feasible TSPTW instance randomly generated using the parameters
        """

        rand = random.Random()

        if seed != -1:
            rand.seed(seed)

        x_coord = [rand.uniform(0, grid_size) for _ in range(n_city)]
        y_coord = [rand.uniform(0, grid_size) for _ in range(n_city)]

        travel_time = []
        time_windows = np.zeros((n_city, 2))

        for i in range(n_city):

            dist = [float(np.sqrt((x_coord[i] - x_coord[j]) ** 2 + (y_coord[i] - y_coord[j]) ** 2))
                    for j in range(n_city)]

            if is_integer_instance:
                dist = [round(x) for x in dist]

            travel_time.append(dist)

        random_solution = list(range(1, n_city))
        rand.shuffle(random_solution)

        random_solution = [0] + random_solution

        time_windows[0, :] = [0, 1000]

        for i in range(1, n_city):

            prev_city = random_solution[i-1]
            cur_city = random_solution[i]

            cur_dist = travel_time[prev_city][cur_city]

            tw_lb_min = time_windows[prev_city, 0] + cur_dist

            rand_tw_lb = rand.uniform(tw_lb_min, tw_lb_min + max_tw_gap)
            rand_tw_ub = rand.uniform(rand_tw_lb, rand_tw_lb + max_tw_size)

            if is_integer_instance:
                rand_tw_lb = np.floor(rand_tw_lb)
                rand_tw_ub = np.ceil(rand_tw_ub)

            time_windows[cur_city, :] = [rand_tw_lb, rand_tw_ub]

        return TSPTW(n_city, travel_time, x_coord, y_coord, time_windows)

    @staticmethod
    def generate_dataset(size, n_city, grid_size, max_tw_gap, max_tw_size, is_integer_instance, seed):
        """
        Generate a dataset of instance
        :param size: the size of the data set
        :param n_city: number of cities
        :param grid_size: x-pos/y-pos of cities will be in the range [0, grid_size]
        :param max_tw_gap: maximum time windows gap allowed between the cities constituing the feasible tour
        :param max_tw_size: time windows of cities will be in the range [0, max_tw_size]
        :param is_integer_instance: True if we want the distances and time widows to have integer values
        :param seed: the seed used for generating the instance
        :return: a dataset of '#size' feasible TSPTW instance randomly generated using the parameters
        """

        dataset = []
        for i in range(size):
            new_instance = TSPTW.generate_random_instance(n_city=n_city, grid_size=grid_size,
                                                          max_tw_gap=max_tw_gap, max_tw_size=max_tw_size,
                                                          is_integer_instance=is_integer_instance, seed=seed)
            dataset.append(new_instance)
            seed += 1

        return dataset
