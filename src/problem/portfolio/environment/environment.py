from src.problem.portfolio.environment.portfolio import *
from src.problem.portfolio.environment.state import *

import torch
import numpy as np

class Environment:

    def __init__(self, instance, n_feat, reward_scaling):
        """
        Initialize the DP/RL environment
        :param instance: a 4-moment portfolio instance
        :param n_feat: number of features for each item
        :param reward_scaling: value for scaling the reward
        """

        self.instance = instance
        self.reward_scaling = reward_scaling
        self.n_feat = n_feat

        self.weigth_norm = np.linalg.norm(self.instance.weights)
        self.mean_norm = np.linalg.norm(self.instance.means)
        self.deviation_norm = np.linalg.norm(self.instance.deviations)
        self.skewnesses_norm = np.linalg.norm(self.instance.skewnesses)
        self.kurtosis_norm = np.linalg.norm(self.instance.kurtosis)

    def get_initial_environment(self):
        """
        Return the initial state of the DP formulation: we are at the city 0 at time 0
        :return: The initial state
        """

        weight = 0
        stage = 0
        item_taken = []

        if self.instance.weights[stage] > self.instance.capacity:
            available_action = set([0])
        else:
            available_action = set([0, 1])

        return State(self.instance, weight, stage, available_action, item_taken)

    def make_nn_input(self, cur_state, mode):
        """
        Return a tensor serving as input of the neural network
        :param cur_state: the current state of the DP model
        :param mode: on GPU or CPU
        :return: the tensor
        """

        feat = [[self.instance.weights[i] / self.weigth_norm,
                      self.instance.means[i] / self.mean_norm,
                      self.instance.deviations[i] / self.deviation_norm,
                      self.instance.skewnesses[i] / self.skewnesses_norm,
                      self.instance.kurtosis[i] / self.kurtosis_norm,
                      (self.instance.capacity - cur_state.weight - self.instance.weights[i]) / self.weigth_norm,
                      0 if i < cur_state.stage else 1, # Remaining
                      1 if i == cur_state.stage else 0,
                      0 if (self.instance.capacity - cur_state.weight -
                            self.instance.weights[i]) >= 0 else 1] # will exceed
                     for i in range(self.instance.n_item)]

        feat_tensor = torch.FloatTensor(feat).reshape(self.instance.n_item, self.n_feat)

        if mode == 'gpu':
            feat_tensor = feat_tensor.cuda()

        return feat_tensor

    def get_next_state_with_reward(self, cur_state, action):
        """
        Compute the next_state and the reward of the RL environment from cur_state when an action is done
        Note that the only non-zero reward is the last one
        :param cur_state: the current state
        :param action: the action that is done
        :return: the next state and the reward collected
        """

        new_state, reward = cur_state.step(action)
        reward = 0
        if new_state.is_done():

            total_mean = sum([new_state.items_taken[i] * self.instance.means[i]
                                   for i in range(self.instance.n_item)])

            total_deviation = sum([new_state.items_taken[i] * self.instance.deviations[i]
                                   for i in range(self.instance.n_item)]) ** (1 / 2)

            total_skewness = sum([new_state.items_taken[i] * self.instance.skewnesses[i]
                                   for i in range(self.instance.n_item)]) ** (1 / 3)

            total_kurtosis = sum([new_state.items_taken[i] * self.instance.kurtosis[i]
                                   for i in range(self.instance.n_item)]) ** (1 / 4)

            reward =  self.instance.moment_factors[0] * total_mean \
                    - self.instance.moment_factors[1] * total_deviation \
                    + self.instance.moment_factors[2] * total_skewness \
                    - self.instance.moment_factors[3] * total_kurtosis

        reward = reward * self.reward_scaling

        return new_state, reward

    def get_valid_actions(self, cur_state):
        """
        Compute and return a binary numpy-vector indicating if the action is still possible or not from the current state
        :param cur_state: the current state of the DP model
        :return: a 1D [0,1]-numpy vector a with a[i] == 1 iff action i is still possible
        """

        action_size = 2

        available = np.zeros(action_size, dtype=np.int)
        available_idx = np.array([x for x in cur_state.available_action], dtype=np.int)
        available[available_idx] = 1
        return available


