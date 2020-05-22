import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt

import time
import sys

import numpy as np
import torch

from src.problem.portfolio.environment.portfolio import Portfolio
from src.problem.portfolio.environment.environment import Environment
from src.problem.portfolio.learning.brain_ppo import BrainPPO
from src.util.replay_memory import ReplayMemory

#  definition of constants
VALIDATION_SET_SIZE = 100
RANDOM_TRIAL = 100
MIN_VAL = -1000000
MAX_VAL = 1000000


class TrainerPPO:
    """
    Definition of the Trainer PPO for the 4-moments Portfolio Optimization problem
    """

    def __init__(self, args):
        """
        Initialization of the trainer
        :param args:  argparse object taking hyperparameters and instance  configuration
        """

        self.args = args
        np.random.seed(self.args.seed)
        self.n_feat = 9
        self.moment_factors = [args.lambda_1, args.lambda_2, args.lambda_3, args.lambda_4]
        self.reward_scaling = 0.001

        self.validation_set = Portfolio.generate_dataset(size=VALIDATION_SET_SIZE,
                                                         n_item=self.args.n_item, lb=0, ub=100,
                                                         capacity_ratio=self.args.capacity_ratio,
                                                         moment_factors=self.moment_factors,
                                                         is_integer_instance=False,
                                                         seed=np.random.randint(10000))

        self.brain = BrainPPO(self.args, self.n_feat)

        self.memory = ReplayMemory()

        self.time_step = 0

        print("***********************************************************")
        print("[INFO] NUMBER OF FEATURES")
        print("[INFO] n_feat: %d" % self.n_feat)
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
                    avg_reward += self.evaluate_instance(j) / self.reward_scaling

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
        instance = Portfolio.generate_random_instance(n_item=self.args.n_item,
                                                      lb=0, ub=100,
                                                      capacity_ratio=self.args.capacity_ratio,
                                                      moment_factors=self.moment_factors,
                                                      is_integer_instance=False,
                                                      seed=-1)

        env = Environment(instance, self.n_feat, self.reward_scaling)

        cur_state = env.get_initial_environment()

        while True:

            self.time_step += 1

            nn_input = env.make_nn_input(cur_state, self.args.mode)
            avail = env.get_valid_actions(cur_state)

            available_tensor = torch.FloatTensor(avail)

            out_action, log_prob_action, _ = self.brain.policy_old.act(nn_input, available_tensor)

            action = out_action.item()
            cur_state, reward = env.get_next_state_with_reward(cur_state, action)

            self.memory.add_sample(nn_input, out_action, log_prob_action, reward, cur_state.is_done(), available_tensor)

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
        env = Environment(instance, self.n_feat, self.reward_scaling)
        cur_state = env.get_initial_environment()

        total_reward = 0

        while True:

            nn_input = env.make_nn_input(cur_state, self.args.mode)
            avail = env.get_valid_actions(cur_state)

            available_tensor = torch.FloatTensor(avail)

            out_action, _, _ = self.brain.policy_old.act(nn_input, available_tensor)

            action = out_action.item()

            cur_state, reward = env.get_next_state_with_reward(cur_state, action)

            total_reward += reward

            if cur_state.is_done():
                break

        return total_reward

