
import random
import os

import torch
import torch.nn as nn

from src.problem.portfolio.learning.actor_critic import ActorCritic


class BrainPPO:
    """
    Definition of the PPO Brain, computing the DQN loss
    """
    def __init__(self, args, n_feat):
        """
        Initialize the PPO Brain
        :param args: argparse object taking hyperparameters
        :param n_feat: number of features on the items
        """
        self.args = args
        self.policy = ActorCritic(self.args, n_feat)
        self.policy_old = ActorCritic(self.args, n_feat)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=args.learning_rate)
        self.MseLoss = nn.MSELoss()

        if args.mode == 'gpu':
            self.policy.cuda()
            self.policy_old.cuda()

    def update(self, memory):
        """
        Compute the loss and update the NN weights through backpropagation of the loss
        :param memory: the replay-memory of samples
        """

        #  accumulated rewards collected on the current episodes for each sample of the memory:
        rewards = []
        acc_reward = 0

        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                acc_reward = 0
            acc_reward = reward + acc_reward
            rewards.insert(0, acc_reward)

        #  Optimize the policy for K epochs:
        for k in range(self.args.k_epochs):

            mem = list(zip(memory.actions, memory.availables, memory.states, memory.log_probs, rewards))
            random.shuffle(mem)
            mem_actions, mem_availables, mem_states, mem_log_probs, mem_rewards = zip(*mem)

            n_batch = self.args.update_timestep // self.args.batch_size

            for j in range(n_batch):

                start_idx = j * self.args.batch_size
                end_idx = (j + 1) * self.args.batch_size - 1

                old_states_for_action = torch.stack(mem_states[start_idx:end_idx])
                old_states_for_value = torch.stack(mem_states[start_idx:end_idx])
                old_actions = torch.stack(mem_actions[start_idx:end_idx])
                old_log_probs = torch.stack(mem_log_probs[start_idx:end_idx])
                old_availables = torch.stack(mem_availables[start_idx:end_idx])
                rewards_tensor = torch.tensor(mem_rewards[start_idx:end_idx])

                if self.args.mode == 'gpu':
                    old_states_for_action.cuda()
                    old_states_for_value.cuda()
                    old_actions = old_actions.cuda()
                    old_log_probs = old_log_probs.cuda()
                    old_availables = old_availables.cuda()
                    rewards_tensor = rewards_tensor.cuda()

                # Evaluating old actions and values
                log_probs, state_values, dist_entropy = self.policy.evaluate(old_states_for_action,
                                                                             old_states_for_value, old_actions,
                                                                             old_availables)

                #  Probability ratio between the old and the new policies
                ratios = torch.exp(log_probs - old_log_probs.detach())

                #  Advantage function
                advantages = rewards_tensor - state_values.detach()

                #  PPO loss value
                surrogate_1 = ratios * advantages
                surrogate_2 = torch.clamp(ratios, 1 - self.args.eps_clip, 1 + self.args.eps_clip) * advantages

                loss = - torch.min(surrogate_1, surrogate_2) + 0.5 * self.MseLoss(state_values, rewards_tensor) \
                       - self.args.entropy_value * dist_entropy

                self.optimizer.zero_grad()

                loss.mean().backward()

                self.optimizer.step()

        #  Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

    def save(self, folder, filename):
        """
        Save the model
        :param folder: Folder requested
        :param filename: file name requested
        """

        filepath = os.path.join(folder, filename)

        if not os.path.exists(folder):
            os.mkdir(folder)
        torch.save(self.policy_old.state_dict(), filepath)
