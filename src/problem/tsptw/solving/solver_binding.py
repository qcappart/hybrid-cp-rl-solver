
import torch
import dgl
import numpy as np
from types import SimpleNamespace

from src.problem.tsptw.environment.tsptw import TSPTW
from src.problem.tsptw.learning.actor_critic import ActorCritic
from src.architecture.graph_attention_network import GATNetwork

class SolverBinding(object):
    """
    Definition of the c++ and the pytorch model.
    """

    def __init__(self, load_folder, n_city, grid_size, max_tw_gap, max_tw_size, seed, rl_algorithm):
        """
        Initialization of the binding
        :param load_folder: folder where the pytorch model (.pth.tar) is saved
        :param n_city: number of cities of the instance
        :param grid_size: x-pos/y-pos of cities will be in the range [0, grid_size]
        :param max_tw_gap: maximum time windows gap allowed between the cities
        :param max_tw_size: time windows of cities will be in the range [0, max_tw_size
        :param seed: seed used for generating the instance
        :param rl_algorithm: 'ppo' or 'dqn'
        """

        self.n_city = n_city
        self.grid_size = grid_size
        self.max_tw_gap = max_tw_gap
        self.max_tw_size = max_tw_size
        self.rl_algorithm = rl_algorithm
        self.seed = seed

        self.max_dist = np.sqrt(self.grid_size ** 2 + self.grid_size ** 2)
        self.max_tw_value = (self.n_city - 1) * (self.max_tw_size + self.max_tw_gap)

        self.instance =  TSPTW.generate_random_instance(n_city=self.n_city, grid_size=self.grid_size,
                                                        max_tw_gap=self.max_tw_gap, max_tw_size=self.max_tw_size,
                                                        seed=seed, is_integer_instance=True)

        self.load_folder = load_folder
        self.model_file, self.latent_dim, self.hidden_layer, self.n_node_feat, self.n_edge_feat = self.find_model()

        self.edge_feat_tensor = self.instance.get_edge_feat_tensor(self.max_dist)

        self.input_graph = self.initialize_graph()

        if self.rl_algorithm == "dqn":

            embedding = [(self.n_node_feat, self.n_edge_feat),
                         (self.latent_dim, self.latent_dim),
                         (self.latent_dim, self.latent_dim),
                         (self.latent_dim, self.latent_dim)]

            self.model = GATNetwork(embedding, self.hidden_layer, self.latent_dim, 1)
            self.model.load_state_dict(torch.load(self.model_file, map_location='cpu'), strict=True)
            self.model.eval()

        elif self.rl_algorithm == "ppo":

            # reproduce the NameSpace of argparse
            args = SimpleNamespace(latent_dim=self.latent_dim, hidden_layer=self.hidden_layer)
            self.actor_critic_network = ActorCritic(args, self.n_node_feat, self.n_edge_feat)

            self.actor_critic_network.load_state_dict(torch.load(self.model_file, map_location='cpu'), strict=True)
            self.model = self.actor_critic_network.action_layer
            self.model.eval()

        else:
            raise Exception("RL algorithm not implemented")

    def get_travel_time_matrix(self):
        """
        Return the travel time/distance matrix (used in the c++ code)
        :return: The travel time matrix
        """
        return self.instance.travel_time

    def get_time_windows_matrix(self):
        """
        Return the time windows matrix (used in the c++ code)
        :return: the time windows matrix
        """

        return self.instance.time_windows

    def initialize_graph(self):
        """
        Return and initialize a graph corresponding to the first state of the DP model.
        :return: the graph
        """

        g = dgl.DGLGraph()
        g.from_networkx(self.instance.graph)

        node_feat = [[self.instance.x_coord[i] / self.grid_size,
                      self.instance.y_coord[i] / self.grid_size,
                      self.instance.time_windows[i][0] / self.max_tw_value,
                      self.instance.time_windows[i][1] / self.max_tw_value,
                      0,
                      1]
                     for i in range(g.number_of_nodes())]

        node_feat_tensor = torch.FloatTensor(node_feat).reshape(g.number_of_nodes(), self.n_node_feat)

        g.ndata['n_feat'] = node_feat_tensor
        g.edata['e_feat'] = self.edge_feat_tensor
        batched_graph = dgl.batch([g])

        return batched_graph

    def find_model(self):
        """
        Find and return the .pth.tar model for the corresponding instance type.
        :return: the location of the model and some hyperparameters used
        """

        log_file_path = self.load_folder + "/log-training.txt"
        best_reward = 0

        best_it = -1

        with open(log_file_path, 'r') as f:
            for line in f:

                if '[INFO]' in line:
                    line = line.split(' ')
                    if line[1] == "latent_dim:":
                        latent_dim = int(line[2].strip())
                    elif line[1] == "hidden_layer:":
                        hidden_layer = int(line[2].strip())
                    elif line[1] == "n_node_feat:":
                        n_node_feat = int(line[2].strip())
                    elif line[1] == "n_edge_feat:":
                        n_edge_feat = int(line[2].strip())

                if '[DATA]' in line:
                    line = line.split(' ')
                    it = int(line[1].strip())
                    reward = float(line[3].strip())
                    if reward > best_reward:
                        best_reward = reward
                        best_it = it

        assert best_it >= 0, "No model found"
        model_str = '%s/iter_%d_model.pth.tar' % (self.load_folder, best_it)
        return model_str, latent_dim, hidden_layer, n_node_feat, n_edge_feat


    def predict_dqn(self, non_fixed_variables, last_visited):
        """
        Given the state related to a node in the CP search, compute the Q-value prediction
        :param non_fixed_variables: variables that are not yet fixed (i.e., must_visit)
        :param last_visited: the last city visited
        :return: the Q-value prediction
        """
        with torch.no_grad():
            self.update_graph_state(non_fixed_variables, last_visited)
            y_pred = self.model(self.input_graph, graph_pooling=False)
            y_pred_tensor = torch.stack([self.input_graph.ndata["n_feat"] for self.input_graph in dgl.unbatch(y_pred)]).squeeze(dim=2)
            y_pred_list = y_pred_tensor.data.cpu().numpy().flatten()

        return y_pred_list

    def predict_ppo(self, non_fixed_variables, last_visited, temperature):
        """
        Given the state related to a node in the CP search, compute the PPO prediction
        :param non_fixed_variables: variables that are not yet fixed (i.e., must_visit)
        :param last_visited: the last city visited
        :param temperature: the softmax temperature for favoring the exploration
        :return: a vector of probabilities of selecting an action
        """

        self.update_graph_state(non_fixed_variables, last_visited)
        y_pred = self.model(self.input_graph, graph_pooling=False)

        out = dgl.unbatch(y_pred)[0]
        action_probs = out.ndata["n_feat"].squeeze(-1)

        available_tensor = torch.zeros([self.n_city])
        available_tensor[non_fixed_variables] = 1

        action_probs = action_probs + torch.abs(torch.min(action_probs))
        action_probs = action_probs - torch.max(action_probs * available_tensor)

        y_pred_list = ActorCritic.masked_softmax(action_probs, available_tensor, dim=0, temperature=temperature)
        y_pred_list = y_pred_list.data.cpu().numpy().flatten()

        return y_pred_list

    def update_graph_state(self, non_fixed_variables, last_visited):
        """
        Update the graph state according to the current state. Must be consistent with the RL environment.
        Especially, mind the normalization constants.
        :param non_fixed_variables: variables that are not yet fixed (i.e., must_visit)
        :param last_visited: the last city visited
        """

        node_feat = [[self.instance.x_coord[i] / self.grid_size,
                      self.instance.y_coord[i] / self.grid_size,
                      self.instance.time_windows[i][0] / self.max_tw_value,
                      self.instance.time_windows[i][1] / self.max_tw_value,
                      0 if i in non_fixed_variables else 1,
                      1 if i == last_visited else 0]
                     for i in range(self.input_graph.number_of_nodes())]

        node_feat_tensor = torch.FloatTensor(node_feat).reshape(self.input_graph.number_of_nodes(), self.n_node_feat)

        self.input_graph.edata['e_feat'] = self.edge_feat_tensor
        self.input_graph.ndata['n_feat'] = node_feat_tensor
        self.input_graph = dgl.batch([self.input_graph])


