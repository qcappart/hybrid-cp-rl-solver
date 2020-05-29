from src.problem.tsptw.environment.tsptw import *
from src.problem.tsptw.environment.state import *

import torch
import numpy as np
import dgl

class Environment:

    def __init__(self, instance, n_node_feat, n_edge_feat, reward_scaling, grid_size, max_tw_gap, max_tw_size):
        """
        Initialize the DP/RL environment
        :param instance: a TSPTW instance
        :param n_node_feat: number of features for the nodes
        :param n_edge_feat: number of features for the edges
        :param reward_scaling: value for scaling the reward
        :param grid_size: x-pos/y-pos of cities will be in the range [0, grid_size] (used for normalization purpose)
        :param max_tw_gap: maximum time windows gap allowed between the cities (used for normalization purpose)
        :param max_tw_size: time windows of cities will be in the range [0, max_tw_size] (used for normalization purpose)
        """

        self.instance = instance
        self.n_node_feat = n_node_feat
        self.n_edge_feat = n_edge_feat
        self.reward_scaling = reward_scaling
        self.grid_size = grid_size
        self.max_tw_gap = max_tw_gap
        self.max_tw_size = max_tw_size

        self.max_dist = np.sqrt(self.grid_size ** 2 + self.grid_size ** 2)
        self.max_tw_value = (self.instance.n_city - 1) * (self.max_tw_size + self.max_tw_gap)
        self.ub_cost = self.max_dist * self.instance.n_city

        self.edge_feat_tensor = self.instance.get_edge_feat_tensor(self.max_dist)

    def get_initial_environment(self):
        """
        Return the initial state of the DP formulation: we are at the city 0 at time 0
        :return: The initial state
        """

        must_visit = set(range(1, self.instance.n_city))  # cities that still have to be visited.
        last_visited = 0  # the current location
        cur_time = 0  # the current time
        cur_tour = [0]  # the tour that is current done

        return State(self.instance, must_visit, last_visited, cur_time, cur_tour)

    def make_nn_input(self, cur_state, mode):
        """
        Return a DGL graph serving as input of the neural network. Assign features on the nodes and on the edges
        :param cur_state: the current state of the DP model
        :param mode: on GPU or CPU
        :return: the DGL graph
        """

        g = dgl.DGLGraph()
        g.from_networkx(self.instance.graph)

        node_feat = [[self.instance.x_coord[i] / self.grid_size,  # x-coord
                      self.instance.y_coord[i] / self.grid_size,  # y-coord
                      self.instance.time_windows[i][0] / self.max_tw_value,  # start of the time windows
                      self.instance.time_windows[i][1] / self.max_tw_value,  # end of the time windows
                      0 if i in cur_state.must_visit else 1,  # 0 if it is possible to visit the node
                      1 if i == cur_state.last_visited else 0]  # 1 if it is the last node visited
                     for i in range(g.number_of_nodes())]

        node_feat_tensor = torch.FloatTensor(node_feat).reshape(g.number_of_nodes(), self.n_node_feat)

        # feeding features into the dgl_graph
        g.ndata['n_feat'] = node_feat_tensor
        g.edata['e_feat'] = self.edge_feat_tensor

        if mode == 'gpu':
            g.ndata['n_feat'] = g.ndata['n_feat'].cuda()
            g.edata['e_feat'] = g.edata['e_feat'].cuda()

        return g

    def get_next_state_with_reward(self, cur_state, action):
        """
        Compute the next_state and the reward of the RL environment from cur_state when an action is done
        :param cur_state: the current state
        :param action: the action that is done
        :return: the next state and the reward collected
        """

        new_state = cur_state.step(action)

        # see the related paper for the reward definition
        reward = self.ub_cost - self.instance.travel_time[cur_state.last_visited][new_state.last_visited]

        if new_state.is_done():
            #  cost of going back to the starting city (always 0)
            reward = reward - self.instance.travel_time[new_state.last_visited][0]


        reward = reward * self.reward_scaling

        return new_state, reward

    def get_valid_actions(self, cur_state):
        """
        Compute and return a binary numpy-vector indicating if the action is still possible or not from the current state
        :param cur_state: the current state of the DP model
        :return: a 1D [0,1]-numpy vector a with a[i] == 1 iff action i is still possible
        """

        available = np.zeros(self.instance.n_city, dtype=np.int)
        available_idx = np.array([x for x in cur_state.must_visit], dtype=np.int)
        available[available_idx] = 1

        return available
