
import torch
import torch.nn as nn
from torch.distributions import Categorical

from src.architecture.set_transformer import SetTransformer

class ActorCritic(nn.Module):
    """
    Definition of an actor-critic architecture used for PPO algorithm.
    """

    def __init__(self, args, n_feat):
        """
        Initialization of the actor-critic with two networks (the actor: outputing probabilities of selecting actions,
        and the critic, outputing an approximation of the state)
        :param args: argparse object taking hyperparameters
        :param n_feat: number of features for each item
        """

        super(ActorCritic, self).__init__()

        self.args = args

        # actor
        self.action_layer = SetTransformer(dim_hidden=args.latent_dim, dim_input=n_feat, dim_output=2)

        # critic
        self.value_layer = SetTransformer(dim_hidden=args.latent_dim, dim_input=n_feat, dim_output=1)


    def act(self, nn_input, available_tensor):
        """
        Perform an action following the probabilities outputed by the current actor
        :param nn_input: the current state, as input for the N
        :param available_tensor: [0,1]-vector of available actions
        :return: the action selection, its log-probability, and its probability
        """

        if self.args.mode == "gpu":
            available_tensor = available_tensor.cuda()

        batched_nn_input = nn_input.unsqueeze(0)

        self.action_layer.eval()
        with torch.no_grad():

            out = self.action_layer(batched_nn_input)
            action_probs = out.squeeze(0)

            #  Doing post-processing on the output to have numerically stable probabilities given that a mask is used
            action_probs = action_probs + torch.abs(torch.min(action_probs))
            action_probs = action_probs - torch.max(action_probs * available_tensor)
            action_probs = self.masked_softmax(action_probs, available_tensor, dim=0)

            dist = Categorical(action_probs)
            action = dist.sample()

        return action, dist.log_prob(action), action_probs

    def evaluate(self, state_for_action, state_for_value, action, available_tensor):
        """
        Evaluating an action wrt. the current policy
        :param state_for_action: State used to compute the actor output
        :param state_for_value: State used to compute the critic output. Although it it the same as the state_for_action,
        it is not the same object
        :param action: the action that is evaluaed
        :param available_tensor: The actions that are possible.
        :return: the log-probabilities of the action, the critic evaluation of the state, the entropy value
        """

        if self.args.mode == "gpu":
            available_tensor = available_tensor.cuda()

        action_probs = self.action_layer(state_for_action)

        action_probs = action_probs + torch.abs(torch.min(action_probs, 1, keepdim=True)[0])
        action_probs = action_probs - torch.max(action_probs * available_tensor, 1, keepdim=True)[0]
        action_probs = self.masked_softmax(action_probs, available_tensor, dim=1)

        dist = Categorical(action_probs)
        action_log_probs = dist.log_prob(action)

        dist_entropy = dist.entropy()
        state_value = self.value_layer(state_for_value)
        return action_log_probs, torch.squeeze(state_value), dist_entropy

    @staticmethod
    def masked_softmax(vector, mask, dim=-1, temperature=1):
        """
        Compute softmax probabilities of a vector having masked values
        :param vector: vector on which the softmax is computed
        :param mask: binary mask for each element of the vector
        :param dim: dimension of the softmax
        :param temperature: temperature for favoring exploration or not
        :return: masked softmax values
        """

        mask_fill_value = -1e32
        memory_efficient = False

        if mask is None:
            result = torch.nn.functional.softmax(vector, dim=dim)
        else:
            mask = mask.float()
            while mask.dim() < vector.dim():
                mask = mask.unsqueeze(1)
            if not memory_efficient:
                # To limit numerical errors from large vector elements outside the mask, we zero these out.
                result = torch.nn.functional.softmax((vector/temperature) * mask, dim=dim)
                result = result * mask
                result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
            else:
                masked_vector = vector.masked_fill((1 - mask).byte(), mask_fill_value)
                result = torch.nn.functional.softmax(masked_vector/temperature, dim=dim)
        return result
