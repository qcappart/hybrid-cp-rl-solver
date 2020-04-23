import random


class ReplayMemory:
    """
    Definition of the replay memory that is used to store sample for PPO
    """

    def __init__(self):
        """
        Initialization of the memory: no sample on it
        """

        self.actions = []
        self.availables = []
        self.states = []
        self.log_probs = []
        self.rewards = []
        self.is_terminals = []

        self.is_shuffled = False

    def add_sample(self, state, action, log_prob, reward, is_terminal, available):
        """
        Add a sample in the memory
        :param state: the current state
        :param action: the action
        :param log_prob: the log-probability of the selected action
        :param reward: the reward collected
        :param is_terminal: is the state terminal ?
        :param available: [0 1] vector indication what are the available actions
        """

        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.is_terminals.append(is_terminal)
        self.availables.append(available)

    def clear_memory(self):
        """
        Clearing the memory: removing all the sample
        """

        del self.actions[:]
        del self.availables[:]
        del self.states[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.is_terminals[:]

        self.is_shuffled = False

    def shuffle(self):
        """
        Randomly shuffle the samples that are on the replay memory
        """

        c = list(zip(self.actions, self.availables, self.states, self.log_probs, self.rewards, self.is_terminals))

        random.shuffle(c)

        actions, availables, states, log_probs, rewards, is_terminals = zip(*c)

        self.actions = list(actions)
        self.availables = list(availables)
        self.states = list(states)
        self.log_probs = list(log_probs)
        self.rewards = list(rewards)
        self.is_terminals = list(is_terminals)

        self.is_shuffled = True
