
import torch
import torch.nn as nn
from torch.distributions import Categorical
import dgl

from src.architecture.graph_attention_network import GATNetwork


class ActorCritic(nn.Module):
    """
    Definition of an actor-critic architecture used for PPO algorithm.
    """

    def __init__(self, args, num_node_feat, num_edge_feat):
        """
        Initialization of the actor-critic with two networks (the actor: outputing probabilities of selecting actions,
        and the critic, outputing an approximation of the state)
        :param args: argparse object taking hyperparameters
        :param num_node_feat: number of features on the nodes
        :param num_edge_feat: numer of features on the edges
        """

        super(ActorCritic, self).__init__()

        self.args = args

        self.embedding = [(num_node_feat, num_edge_feat),
                          (self.args.latent_dim, self.args.latent_dim),
                          (self.args.latent_dim, self.args.latent_dim),
                          (self.args.latent_dim, self.args.latent_dim),
                          (self.args.latent_dim, self.args.latent_dim)]

        # actor
        self.action_layer = GATNetwork(self.embedding, self.args.hidden_layer, self.args.latent_dim, 1)

        # critic
        self.value_layer = GATNetwork(self.embedding, self.args.hidden_layer, self.args.latent_dim, 1)

    def act(self, graph_state, available_tensor):
        """
        Perform an action following the probabilities outputed by the current actor
        :param graph_state: the current state
        :param available_tensor: [0,1]-vector of available actions
        :return: the action selection, its log-probability, and its probability
        """

        if self.args.mode == "gpu":
            available_tensor = available_tensor.cuda()

        batched_graph = dgl.batch([graph_state, ])

        self.action_layer.eval()
        with torch.no_grad():
            out = self.action_layer(batched_graph, graph_pooling=False)
            out = dgl.unbatch(out)[0]
            action_probs = out.ndata["n_feat"].squeeze(-1)

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

        out = self.action_layer(state_for_action, graph_pooling=False)

        out = [x.ndata["n_feat"] for x in dgl.unbatch(out)]

        action_probs = torch.stack(out).squeeze(-1)
        action_probs = action_probs + torch.abs(torch.min(action_probs, 1, keepdim=True)[0])
        action_probs = action_probs - torch.max(action_probs * available_tensor, 1, keepdim=True)[0]
        action_probs = self.masked_softmax(action_probs, available_tensor, dim=1)

        dist = Categorical(action_probs)
        action_log_probs = dist.log_prob(action)

        dist_entropy = dist.entropy()
        state_value = self.value_layer(state_for_value, graph_pooling=True)
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
