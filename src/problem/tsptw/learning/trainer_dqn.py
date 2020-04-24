import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt

import random
import time
import sys

import numpy as np

import dgl

from src.problem.tsptw.environment.environment import Environment
from src.problem.tsptw.learning.brain_dqn import BrainDQN
from src.problem.tsptw.environment.tsptw import TSPTW
from src.util.prioritized_replay_memory import PrioritizedReplayMemory

#  definition of constants
MEMORY_CAPACITY = 50000
GAMMA = 1
STEP_EPSILON = 5000.0
UPDATE_TARGET_FREQUENCY = 500
VALIDATION_SET_SIZE = 100
RANDOM_TRIAL = 100
MAX_BETA = 10
MIN_VAL = -1000000
MAX_VAL = 1000000


class TrainerDQN:
    """
    Definition of the Trainer DQN for the TSPTW
    """

    def __init__(self, args):
        """
        Initialization of the trainer
        :param args:  argparse object taking hyperparameters and instance  configuration
        """

        self.args = args
        np.random.seed(self.args.seed)
        self.instance_size = self.args.n_city
        self.n_action = self.instance_size - 1  # Because we begin at a given city, so we have 1 city less to visit

        self.num_node_feats = 6
        self.num_edge_feats = 5

        self.reward_scaling = 0.001

        self.validation_set = TSPTW.generate_dataset(size=VALIDATION_SET_SIZE, n_city=self.args.n_city,
                                                     grid_size=self.args.grid_size, max_tw_gap=self.args.max_tw_gap,
                                                     max_tw_size=self.args.max_tw_size, is_integer_instance=False,
                                                     seed=np.random.randint(10000))

        self.brain = BrainDQN(self.args, self.num_node_feats, self.num_edge_feats)
        self.memory = PrioritizedReplayMemory(MEMORY_CAPACITY)

        self.steps_done = 0
        self.init_memory_counter = 0

        if args.n_step == -1: # We go until the end of the episode
            self.n_step = self.n_action
        else:
            self.n_step = self.args.n_step

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

        self.initialize_memory()

        print('[INFO]', 'iter', 'time', 'avg_reward_learning', 'loss', "beta")

        cur_best_reward = MIN_VAL

        for i in range(self.args.n_episode):

            loss, beta = self.run_episode(i, memory_initialization=False)

            #  We first evaluate the validation step every 10 episodes, until 100, then every 100 episodes.
            if (i % 10 == 0 and i < 101) or i % 100 == 0:

                avg_reward = 0.0
                for j in range(len(self.validation_set)):
                    avg_reward += self.evaluate_instance(j)

                avg_reward = avg_reward / len(self.validation_set)

                cur_time = round(time.time() - start_time, 2)

                print('[DATA]', i, cur_time, avg_reward, loss, beta)

                sys.stdout.flush()

                if self.args.plot_training:
                    iter_list.append(i)
                    reward_list.append(avg_reward)
                    plt.clf()

                    plt.plot(iter_list, reward_list, linestyle="-", label="DQN", color='y')

                    plt.legend(loc=3)
                    out_file = '%s/training_curve_reward.png' % self.args.save_dir
                    plt.savefig(out_file)

                fn = "iter_%d_model.pth.tar" % i

                #  We record only the model that is better on the validation set wrt. the previous model
                #  We nevertheless record a model every 10000 episodes
                if avg_reward >= cur_best_reward:
                    cur_best_reward = avg_reward
                    self.brain.save(folder=self.args.save_dir, filename=fn)
                elif i % 10000 == 0:
                    self.brain.save(folder=self.args.save_dir, filename=fn)

    def initialize_memory(self):
        """
        Initialize the replay memory with random episodes and a random selection
        """

        while self.init_memory_counter < MEMORY_CAPACITY:
            self.run_episode(0, memory_initialization=True)

        print("[INFO] Memory Initialized")

    def run_episode(self, episode_idx, memory_initialization):
        """
        Run a single episode, either for initializing the memory (random episode in this case)
        or for training the model (following DQN algorithm)
        :param episode_idx: the index of the current episode done (without considering the memory initialization)
        :param memory_initialization: True if it is for initializing the memory
        :return: the loss and the current beta of the softmax selection
        """

        #  Generate a random instance
        instance = TSPTW.generate_random_instance(n_city=self.args.n_city, grid_size=self.args.grid_size,
                                                  max_tw_gap=self.args.max_tw_gap, max_tw_size=self.args.max_tw_size,
                                                  seed=-1, is_integer_instance=False)

        env = Environment(instance, self.num_node_feats, self.num_edge_feats, self.reward_scaling,
                          self.args.grid_size, self.args.max_tw_gap, self.args.max_tw_size)

        cur_state = env.get_initial_environment()

        graph_list = [dgl.DGLGraph()] * self.n_action
        rewards_vector = np.zeros(self.n_action)
        actions_vector = np.zeros(self.n_action, dtype=np.int16)
        available_vector = np.zeros((self.n_action, self.args.n_city))

        idx = 0
        total_loss = 0

        #  the current temperature for the softmax selection: increase from 0 to MAX_BETA
        temperature = max(0., min(self.args.max_softmax_beta,
                                  (episode_idx - 1) / STEP_EPSILON * self.args.max_softmax_beta))

        #  execute the episode
        while True:

            graph = env.make_nn_input(cur_state, self.args.mode)
            avail = env.get_valid_actions(cur_state)
            avail_idx = np.argwhere(avail == 1).reshape(-1)

            if memory_initialization:  # if we are in the memory initialization phase, a random episode is selected
                action = random.choice(avail_idx)
            else:  # otherwise, we do the softmax selection
                action = self.soft_select_action(graph, avail, temperature)

                #  each time we do a step, we increase the counter, and we periodically synchronize the target network
                self.steps_done += 1
                if self.steps_done % UPDATE_TARGET_FREQUENCY == 0:
                    self.brain.update_target_model()

            cur_state, reward = env.get_next_state_with_reward(cur_state, action)

            graph_list[idx] = graph
            rewards_vector[idx] = reward
            actions_vector[idx] = action
            available_vector[idx] = avail

            if cur_state.is_done():
                break

            idx += 1

        episode_last_idx = idx

        #  compute the n-step values
        for i in range(self.n_action):

            if i <= episode_last_idx:
                cur_graph = graph_list[i]
                cur_available = available_vector[i]
            else:
                cur_graph = graph_list[episode_last_idx]
                cur_available = available_vector[episode_last_idx]

            if i + self.n_step < self.n_action:
                next_graph = graph_list[i + self.n_step]
                next_available = available_vector[i + self.n_step]
            else:
                next_graph = dgl.DGLGraph()
                next_available = env.get_valid_actions(cur_state)

            #  a state correspond to the graph, with the nodes that we can still visit
            state_features = (cur_graph, cur_available)
            next_state_features = (next_graph, next_available)

            #  the n-step reward
            reward = sum(rewards_vector[i:i+self.n_step])
            action = actions_vector[i]

            sample = (state_features, action, reward, next_state_features)

            if memory_initialization:
                error = abs(reward)  # the error of the replay memory is equals to the reward, at initialization
                self.init_memory_counter += 1
                step_loss = 0
            else:
                x, y, errors = self.get_targets([(0, sample, 0)])  # feed the memory with the new samples
                error = errors[0]
                step_loss = self.learning()  # learning procedure

            self.memory.add(error, sample)

            total_loss += step_loss

        return total_loss, temperature

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

            action = self.select_action(graph, avail)

            cur_state, reward = env.get_next_state_with_reward(cur_state, action)

            total_reward += reward

            if cur_state.is_done():
                break

        return total_reward

    def select_action(self, graph, available):
        """
        Select an action according the to the current model
        :param graph: the graph (first part of the state)
        :param available: the vector of available (second part of the state)
        :return: the action, following the greedy policy with the model prediction
        """

        batched_graph = dgl.batch([graph, ])
        available = available.astype(bool)
        out = self.brain.predict(batched_graph, target=False)[0].reshape(-1)

        action_idx = np.argmax(out[available])

        action = np.arange(len(out))[available][action_idx]

        return action

    def soft_select_action(self, graph, available, beta):
        """
        Select an action according the to the current model with a softmax selection of temperature beta
        :param graph: the graph (first part of the state)
        :param available: the vector of available (second part of the state)
        :param beta: the current temperature
        :return: the action, following the softmax selection with the model prediction
        """

        batched_graph = dgl.batch([graph, ])
        available = available.astype(bool)
        out = self.brain.predict(batched_graph, target=False)[0].reshape(-1)

        if len(out[available]) > 1:
            logits = (out[available] - out[available].mean())
            div = ((logits ** 2).sum() / (len(logits) - 1)) ** 0.5
            logits = logits / div

            probabilities = np.exp(beta * logits)
            norm = probabilities.sum()

            if norm == np.infty:
                action_idx = np.argmax(logits)
                action = np.arange(len(out))[available][action_idx]
                return action, 1.0

            probabilities /= norm
        else:
            probabilities = [1]

        action_idx = np.random.choice(np.arange(len(probabilities)), p=probabilities)
        action = np.arange(len(out))[available][action_idx]
        return action

    def get_targets(self, batch):
        """
        Compute the TD-errors using the n-step Q-learning function and the model prediction
        :param batch: the batch to process
        :return: the state input, the true y, and the error for updating the memory replay
        """

        batch_len = len(batch)
        graph, avail = list(zip(*[e[1][0] for e in batch]))

        graph_batch = dgl.batch(graph)

        next_graph, next_avail = list(zip(*[e[1][3] for e in batch]))
        next_graph_batch = dgl.batch(next_graph)
        next_copy_graph_batch = dgl.batch(dgl.unbatch(next_graph_batch))
        p = self.brain.predict(graph_batch, target=False)

        if next_graph_batch.number_of_nodes() > 0:

            p_ = self.brain.predict(next_graph_batch, target=False)
            p_target_ = self.brain.predict(next_copy_graph_batch, target=True)

        x = []
        y = []
        errors = np.zeros(len(batch))

        for i in range(batch_len):

            sample = batch[i][1]
            state_graph, state_avail = sample[0]
            action = sample[1]
            reward = sample[2]
            next_state_graph, next_state_avail = sample[3]
            next_action_indices = np.argwhere(next_state_avail == 1).reshape(-1)
            t = p[i]

            q_value_prediction = t[action]

            if len(next_action_indices) == 0:

                td_q_value = reward
                t[action] = td_q_value

            else:

                mask = np.zeros(p_[i].shape, dtype=bool)
                mask[next_action_indices] = True

                best_valid_next_action_id = np.argmax(p_[i][mask])
                best_valid_next_action = np.arange(len(mask))[mask.reshape(-1)][best_valid_next_action_id]

                td_q_value = reward + GAMMA * p_target_[i][best_valid_next_action]
                t[action] = td_q_value

            state = (state_graph, state_avail)
            x.append(state)
            y.append(t)

            errors[i] = abs(q_value_prediction - td_q_value)

        return x, y, errors

    def learning(self):
        """
        execute a learning step on a batch of randomly selected experiences from the memory
        :return: the subsequent loss
        """

        batch = self.memory.sample(self.args.batch_size)

        x, y, errors = self.get_targets(batch)

        #  update the errors
        for i in range(len(batch)):
            idx = batch[i][0]
            self.memory.update(idx, errors[i])

        loss = self.brain.train(x, y)

        return round(loss, 4)



