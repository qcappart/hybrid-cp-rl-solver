import networkx as nx
import random
import heapq
import numpy as np
import torch


def abs_norm(x):
    """
    return the 1-norm of x
    """
    return np.linalg.norm(x, ord=1)
class PDTSP:

    def __init__(self, n_city, m_commodity, distances, x_coord, y_coord, pickup_constraints, capacity):
        """
        Create an instance of the TSPTW problem
        :param n_city: number of cities
        :param distances: distance matrix between the cities
        :param x_coord: list of x-pos of the cities
        :param y_coord: list of y-pos of the cities
        :param pickup_constraints: list of commodity vectors for each city ([weight_0, ..., weight_m] for each city, negative if delivered)
        """

        self.n_city = n_city
        self.m_commodity = m_commodity
        self.distances = distances
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.pickup_constraints = pickup_constraints
        self.capacity = capacity
        self.graph = self.build_graph()

    def build_graph(self):
        """
        Build a networkX graph representing the PDTSP instance. Features on the edges are the distances
        and 4 binary values stating if the edge is part of the (1, 5, 10, 20) nearest neighbors of a node?
        :return: the graph
        """

        g = nx.DiGraph()

        for i in range(self.n_city):

            cur_distances = self.distances[i][:]

            # +1 because we remove the self-edge (cost 0)
            k_min_idx_1 = heapq.nsmallest(1 + 1, range(len(cur_distances)), cur_distances.__getitem__)
            k_min_idx_5 = heapq.nsmallest(5 + 1, range(len(cur_distances)), cur_distances.__getitem__)
            k_min_idx_10 = heapq.nsmallest(10 + 1, range(len(cur_distances)), cur_distances.__getitem__)
            k_min_idx_20 = heapq.nsmallest(20 + 1, range(len(cur_distances)), cur_distances.__getitem__)

            for j in range(self.n_city):

                if i != j:

                    is_k_neigh_1 = 1 if j in k_min_idx_1 else 0
                    is_k_neigh_5 = 1 if j in k_min_idx_5 else 0
                    is_k_neigh_10 = 1 if j in k_min_idx_10 else 0
                    is_k_neigh_20 = 1 if j in k_min_idx_20 else 0

                    weight = self.distances[i][j]
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
    def generate_random_instance(n_city, m_commodity, grid_size, max_commodity_weight, capacity_looseness,
                                 seed, is_integer_instance):
        """
        :param n_city: number of cities
        :param m_commoditiy: number of different commodities
        :param grid_size: x-pos/y-pos of cities will be in the range [0, grid_size]
        :param max_commodity_weight: max weight of a single commodity
        :param capacity_looseness: max unused capacity of the vehicle
        :param seed: seed used for generating the instance. -1 means no seed (instance is random)
        :param is_integer_instance: True if we want the distances and time widows to have integer values
        :return: a feasible TSPTW instance randomly generated using the parameters.
        """
        
        rand = random.Random()

        if seed != -1:
            rand.seed(seed)

        x_coord = [rand.uniform(0, grid_size) for _ in range(n_city)]
        y_coord = [rand.uniform(0, grid_size) for _ in range(n_city)]

        distances = []
        pickup_constraints = np.zeros((n_city, m_commodity))

        for i in range(n_city):
            dist = [float(np.sqrt((x_coord[i] - x_coord[j]) ** 2 + (y_coord[i] - y_coord[j]) ** 2))
                    for j in range(n_city)]

            if is_integer_instance:
                dist = [round(x) for x in dist]

            distances.append(dist)

        random_solution = list(range(1, n_city))
        rand.shuffle(random_solution)

        random_solution = [0] + random_solution

        pickup_constraints[0, :] = [0] * m_commodity  # no pick up or delivery at the depot

        random_pairings = [rand.sample(range(n_city), 2) for _ in range(m_commodity)]
        random_pairings = [(random_solution[min(pairing)], random_solution[max(pairing)]) for pairing in random_pairings]  # ensures an acyclic graph

        for k, (pick_up_city, delivery_city) in enumerate(random_pairings):
            commodity_weight = random.uniform(0, max_commodity_weight)
            if is_integer_instance:
                commodity_weight = int(commodity_weight)
            weight_vector = np.zeros(m_commodity)
            weight_vector[k] = commodity_weight
            pickup_constraints[pick_up_city] += weight_vector
            pickup_constraints[delivery_city] -= weight_vector
        
        cur_load = np.zeros(m_commodity)
        max_load = -1
        for i in range(1, n_city):
            cur_load += pickup_constraints[i]
            total_load = np.linalg.norm(cur_load, ord=1)
            max_load = max(max_load, total_load)
        
        capacity = rand.uniform(max_load, max_load + capacity_looseness)
        return PDTSP(n_city, m_commodity, distances, x_coord, y_coord, pickup_constraints, capacity)

    @staticmethod
    def generate_dataset(size, n_city, m_commodity, grid_size, max_commodity_weight, capacity_looseness,
                                 seed, is_integer_instance):
        """
        Generate a dataset of instance
        :param size: the size of the data set
        :param seed: the seed used for generating the instance
        :param n_city: number of cities
        :param grid_size: x-pos/y-pos of cities will be in the range [0, grid_size]
        :param max_commodity_weight: maximum single commodity weight.
        :param capacity_looseness: maximum unused capacity in the feasible tour
        :param is_integer_instance: True if we want the distances and commodities weights to have integer values
        :return: a dataset of '#size' feasible PDTSP instance randomly generated using the parameters.
        """

        dataset = []
        for _ in range(size):
            new_instance = PDTSP.generate_random_instance(n_city, m_commodity, grid_size, max_commodity_weight, capacity_looseness,
                                 seed, is_integer_instance)
            dataset.append(new_instance)
            seed += 1

        return dataset

if __name__ == '__main__':
    instance = PDTSP.generate_random_instance(10, 3, 10, 1, 2, 0, False)
    print('done')