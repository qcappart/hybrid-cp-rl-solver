"""
Graph Attention Networks in DGL whit nodes and edges features.

Adaptation of:
----------
Paper: https://arxiv.org/abs/1710.10903
Author's src: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

import torch
import torch.nn as nn
import dgl
import numpy as np

SQRT_TWO = float(np.sqrt(2))


class EdgeFtLayer(nn.Module):
    """
    Definition of a GAT message passing layer with fetures on the edges
    """

    def __init__(self, v_in_dim, v_out_dim, e_in_dim, e_out_dim):
        """
        Initilization of the layer
        :param v_in_dim: input dimension of the nodes
        :param v_out_dim: output dimension of the nodes
        :param e_in_dim: input dimension of the edges
        :param e_out_dim: output dimension of the edges
        """

        super(self.__class__, self).__init__()
        self.v_in_dim = v_in_dim
        self.v_out_dim = v_out_dim
        self.e_in_dim = e_in_dim
        self.e_out_dim = e_out_dim

        #  Node feature transformation weights
        self.W_a = nn.Parameter(nn.init.xavier_normal_(torch.zeros((2 * v_in_dim + e_in_dim, v_out_dim)),
                                                       gain=SQRT_TWO))
        self.W_T = nn.Parameter(nn.init.xavier_normal_(torch.zeros((2 * v_in_dim + e_in_dim, v_out_dim)),
                                                       gain=SQRT_TWO))

        self.b_T = nn.Parameter(torch.zeros(v_out_dim))

        # Edge feature transformation weights
        self.W_e = nn.Parameter(nn.init.xavier_normal_(torch.zeros((v_in_dim, e_out_dim)), gain=SQRT_TWO))
        self.W_ee = nn.Parameter(nn.init.xavier_normal_(torch.zeros((e_in_dim, e_out_dim)), gain=SQRT_TWO))

        self.prelu = nn.PReLU()

    def edge_function(self, edges):
        """
        Compute the edge weights according to the neighbor nodes.
        :param edges: the edges of the graphs
        :return: the weights
        """

        node_src = edges.src['n_feat']
        node_dest = edges.dst['n_feat']
        e = edges.data['e_feat']
        new_e_feat = torch.matmul(node_src, self.W_e) + torch.matmul(node_dest, self.W_e) + torch.matmul(e, self.W_ee)
        return {'e_feat': new_e_feat}

    def message_function(self, edges):
        """
        The message passing function with attention.
        :param edges:  the edges of the graphs
        :return:  the attention logits and the node weights (without the attention)
        """

        N1 = edges.src['n_feat']
        N2 = edges.dst['n_feat']
        e = edges.data['e_feat']
        x = torch.cat([N2, e, N1], dim=1)

        attention_logits = self.prelu(torch.matmul(x, self.W_a))
        unattended_node_features = torch.matmul(x, self.W_T)

        return {'attention_logits': attention_logits,
                "unattended_node_features": unattended_node_features
                }

    def new_node_features(self, nodes):
        """
        The reduction function with attention
        :param nodes: the nodes of the graphs
        :return: the weights of the nodes after the attention application
        """

        attention = torch.softmax(nodes.mailbox["attention_logits"], dim=1)
        unattended_features = nodes.mailbox["unattended_node_features"]

        new_features = torch.sum(attention * unattended_features, dim=1) + self.b_T
        return {"new_n_feat": new_features}

    def forward(self, g):
        """
        Forward pass of the edge layer. Note that the graph wfeatures are updated.
        :param g: the graph
        :return: the updated graph
        """
        g.update_all(message_func=self.message_function, reduce_func=self.new_node_features)
        g.apply_edges(self.edge_function)
        g.ndata["n_feat"] = g.ndata.pop("new_n_feat")
        return g


class GATNetwork(nn.Module):
    """
    Definition of the GAT neural network
    """

    def __init__(self, layer_features, n_hidden_layer, latent_dim, output_dim):
        """
        Initialization of the networks
        :param layer_features: list [(n_feat_1, e_feat_1),...] of the dimension of the GAT embedding
        :param n_hidden_layer: Number of hidden layers in the feedforward network after the embedding
        :param latent_dim: Number of nodes in each feedforward layers after the embedding
        :param output_dim: Output dimension of the network
        """

        super(self.__class__, self).__init__()
        self.n_dim, self.e_dim = zip(*layer_features)
        self.n_layers = len(self.n_dim) - 1
        self.output_dim = output_dim
        self.embedding_layer = []

        for l in range(0, self.n_layers):
            layer = EdgeFtLayer(self.n_dim[l], self.n_dim[l+1], self.e_dim[l], self.e_dim[l+1])
            self.embedding_layer.append(layer)

        self.embedding_layer = nn.ModuleList(self.embedding_layer)

        out_dim = layer_features[-1][0]  # nodes features of the last layer

        self.fc_layer = []
        self.fc_layer.append(nn.Linear(out_dim, latent_dim))

        for i in range(n_hidden_layer):
            self.fc_layer.append(nn.Linear(latent_dim, latent_dim))

        self.fc_layer = nn.ModuleList(self.fc_layer)
        self.fc_out = nn.Linear(latent_dim, self.output_dim)

    def forward(self, g, graph_pooling):
        """
        Forward pass on the graph.
        :param g: The graph
        :param graph_pooling: Binary value indicating if the GAT embedding must be pooled (with max) in order to have
        an output on the global graph. Otherwise, output are node-dependant
        :return: prediction of the GAT network
        """

        for l, layer in enumerate(self.embedding_layer):
            g = layer(g)
            g.ndata["n_feat"] = torch.relu(g.ndata["n_feat"])
            g.edata["e_feat"] = torch.relu(g.edata["e_feat"])

        if graph_pooling:
            out = dgl.max_nodes(g, "n_feat")
            for l, layer in enumerate(self.fc_layer):
                out = torch.relu(layer(out))
            out = self.fc_out(out)
            return out

        else:
            for l, layer in enumerate(self.fc_layer):
                g.ndata["n_feat"] = torch.relu(layer(g.ndata["n_feat"]))
            g.ndata["n_feat"] = self.fc_out(g.ndata["n_feat"])

            return g