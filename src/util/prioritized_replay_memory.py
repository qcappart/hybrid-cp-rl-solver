
import random
import numpy as np

from src.util.sum_tree import SumTree


class PrioritizedReplayMemory:
    """
    Definition of a prioritized replay memory that is used to store sample for DQN
    Based on the following implementation:
    https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/
    https://arxiv.org/abs/1511.05952
    """

    e = 0.01
    a = 0.6
    b = 0.4
    b_increment = 0.001

    def __init__(self, capacity):
        """
        Initializing a memory of a given capacity
        :param capacity: the capacity of the memory
        """

        self.tree = SumTree(capacity)
        self.capacity = capacity

    def get_priority(self, error):
        """
        Compute the probability related to a given error
        :param error: the distance between the Q-value (Q(s,a) and its target T(S)
        :return: the probability of sampling the experience
        """

        sampling_prob = (error + self.e) ** self.a
        return sampling_prob

    def add(self, error, sample):
        """
        Add a new sample in the memory
        :param error:  the error of the sample
        :param sample: the sample (s, a, r, s)
        """

        p = self.get_priority(error)
        self.tree.add(p, sample)

    def get_importance_sampling_weight(self, p):
        """
        Compute the importance sampling weight of a probability
        :param p: the probability
        :return: the importance sampling weight
        """

        return (p * self.capacity) ** (-self.b)  # equivalent to ((1./self.capacity) * (1./p)) ** self.b

    def sample(self, n):
        """
        Sample n experiences of the memory
        :param n: the number os samples
        :return: the batch of sampled experiences
        """

        batch = []
        segment = self.tree.total() / n
        self.b = min(1, self.b + self.b_increment)

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total()
        max_weight = self.get_importance_sampling_weight(min_prob)

        for i in range(n):
            lb = segment * i
            ub = segment * (i + 1)
            s = random.uniform(lb, ub)
            (idx, p, data) = self.tree.get(s)
            sampling_prob = p / self.tree.total()
            weight_importance_sampling = self.get_importance_sampling_weight(sampling_prob) / max_weight

            batch.append((idx, data, weight_importance_sampling))

        return batch

    def update(self, idx, error):
        """
        Update the probability of a sample with the given error
        :param idx: the index of the sample
        :param error: the error
        """

        p = self.get_priority(error)
        self.tree.update(idx, p)
