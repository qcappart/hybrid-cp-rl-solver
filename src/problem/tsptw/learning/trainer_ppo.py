import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt

import time
import sys

import numpy as np
import torch
import dgl

from src.problem.tsptw.environment.tsptw import TSPTW
from src.problem.tsptw.environment.environment import Environment
from src.problem.tsptw.learning.brain_ppo import BrainPPO
from src.util.replay_memory import ReplayMemory

#  definition of constants
VALIDATION_SET_SIZE = 100
RANDOM_TRIAL = 100
MIN_VAL = -1000000
MAX_VAL = 1000000


class TrainerPPO:
    """
    Definition of the Trainer PPO for the TSPTW
    """

    def __init__(self, args):
        """
        Initialization of the trainer
        :param args:  argparse object taking hyperparameters and instance  configuration
        """

        self.args = args
        np.random.seed(self.args.seed)
        self.num_node_feats = 6
        self.num_edge_feats = 5

        self.reward_scaling = 0.001

        self.validation_set = TSPTW.generate_dataset(size=VALIDATION_SET_SIZE, n_city=self.args.n_city,
                                                     grid_size=self.args.grid_size, max_tw_gap=self.args.max_tw_gap,
                                                     max_tw_size=self.args.max_tw_size, is_integer_instance=False,
                                                     seed=np.random.randint(10000))

        self.brain = BrainPPO(self.args, self.num_node_feats, self.num_edge_feats)

        self.memory = ReplayMemory()

        self.time_step = 0

        print("***********************************************************")
        print("[INFO] NUMBER OF FEATURES")
        print("[INFO] n_node_feat: %d" % self.num_node_feats)
        print("[INFO] n_edge_feat: %d" % self.num_edge_feats)
        print("***********************************************************")

    def run_training(self):
        """
        Run de main loop for training the model
        """

        start_time = time.time()

        if self.args.plot_training:
            iter_list = []
            reward_list = []

        print('[INFO]', 'iter', 'time', 'avg_reward_learning')

        cur_best_reward = MIN_VAL

        for i in range(self.args.n_episode):

            self.run_episode()

            #  We first evaluate the validation step every 10 episodes, until 100, then every 100 episodes.
            if (i % 10 == 0 and i < 101) or i % 100 == 0:

                avg_reward = 0.0
                for j in range(len(self.validation_set)):
                    avg_reward += self.evaluate_instance(j)

                avg_reward = avg_reward / len(self.validation_set)

                cur_time = round(time.time() - start_time, 2)

                print('[DATA]', i, cur_time, avg_reward)

                sys.stdout.flush()

                if self.args.plot_training:
                    iter_list.append(i)
                    reward_list.append(avg_reward)
                    plt.clf()
                    plt.plot(iter_list, reward_list, linestyle="-", label="PPO", color='y')

                    plt.legend(loc=3)
                    out_file = '%s/training_curve_reward.png' % self.args.save_dir
                    plt.savefig(out_file)
                    plt.clf()

                fn = "iter_%d_model.pth.tar" % i

                #  We record only the model that is better on the validation set wrt. the previous model
                #  We nevertheless record a model every 10000 episodes
                if avg_reward >= cur_best_reward:
                    cur_best_reward = avg_reward
                    self.brain.save(self.args.save_dir, fn)
                elif i % 10000 == 0:
                    self.brain.save(self.args.save_dir, fn)

    def run_episode(self):
        """
        Run the training for a single episode
        """

        #  Generate a random instance
        instance = TSPTW.generate_random_instance(n_city=self.args.n_city, grid_size=self.args.grid_size,
                                                  max_tw_gap=self.args.max_tw_gap, max_tw_size=self.args.max_tw_size,
                                                  seed=-1, is_integer_instance=False)

        env = Environment(instance, self.num_node_feats, self.num_edge_feats, self.reward_scaling,
                          self.args.grid_size, self.args.max_tw_gap, self.args.max_tw_size)

        cur_state = env.get_initial_environment()

        while True:

            self.time_step += 1

            graph = env.make_nn_input(cur_state, self.args.mode)
            avail = env.get_valid_actions(cur_state)

            available_tensor = torch.FloatTensor(avail)

            out_action, log_prob_action, _ = self.brain.policy_old.act(graph, available_tensor)

            action = out_action.item()
            cur_state, reward = env.get_next_state_with_reward(cur_state, action)

            self.memory.add_sample(graph, out_action, log_prob_action, reward, cur_state.is_done(), available_tensor)

            if self.time_step % self.args.update_timestep == 0:
                self.brain.update(self.memory)
                self.memory.clear_memory()
                self.time_step = 0

            if cur_state.is_done():
                break

    def evaluate_instance(self, idx):
        """
        Evaluate an instance with the current model
        :param idx: the index of the instance in the validation set
        :return: the reward collected for this instance
        """

        instance = self.validation_set[idx]
        env = Environment(instance, self.num_node_feats, self.num_edge_feats, self.reward_scaling,
                          self.args.grid_size, self.args.max_tw_gap, self.args.max_tw_size)
        cur_state = env.get_initial_environment()

        total_reward = 0

        while True:

            graph = env.make_nn_input(cur_state, self.args.mode)
            avail = env.get_valid_actions(cur_state)

            available_tensor = torch.FloatTensor(avail)

            batched_graph = dgl.batch([graph, ])

            out_action, _, _ = self.brain.policy_old.act(batched_graph, available_tensor)

            action = out_action.item()

            cur_state, reward = env.get_next_state_with_reward(cur_state, action)

            total_reward += reward

            if cur_state.is_done():
                break

        return total_reward

