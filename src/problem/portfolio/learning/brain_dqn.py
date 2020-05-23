
import torch
import torch.optim as optim
import torch.nn.functional as F

import os
import numpy as np

from src.architecture.set_transformer import SetTransformer


class BrainDQN:
    """
    Definition of the DQN Brain, computing the DQN loss
    """

    def __init__(self, args, n_feat):
        """
        Initialization of the DQN Brain
        :param args: argparse object taking hyperparameters
        :param n_feat: number of features on the items
        """

        self.args = args

        self.model = SetTransformer(dim_hidden=args.latent_dim, dim_input=n_feat, dim_output=2)
        self.target_model = SetTransformer(dim_hidden=args.latent_dim, dim_input=n_feat, dim_output=2)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

        if self.args.mode == 'gpu':
            self.model.cuda()
            self.target_model.cuda()

    def train(self, x, y):
        """
        Compute the loss between (f(x) and y)
        :param x: the input
        :param y: the true value of y
        :return: the loss
        """

        self.model.train()

        set_input, _ = list(zip(*x))
        batched_set = torch.stack(set_input)

        y_pred = self.model(batched_set)
        y_tensor = torch.FloatTensor(np.array(y))

        if self.args.mode == 'gpu':
            y_tensor = y_tensor.contiguous().cuda()

        loss = F.smooth_l1_loss(y_pred, y_tensor)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def predict(self, nn_input, target):
        """
        Predict the Q-values using the current state, either using the model or the target model
        :param nn_input: the featurized state
        :param target: True is the target network must be used for the prediction
        :return: A list of the predictions for each node
        """

        with torch.no_grad():

            if target:
                self.target_model.eval()
                res = self.target_model(nn_input)
            else:
                self.model.eval()
                res = self.model(nn_input)

        return res.cpu().numpy()

    def update_target_model(self):
        """
        Synchronise the target network with the current one
        """

        self.target_model.load_state_dict(self.model.state_dict())

    def save(self, folder, filename):
        """
        Save the model
        :param folder: Folder requested
        :param filename: file name requested
        """

        filepath = os.path.join(folder, filename)

        if not os.path.exists(folder):
            os.mkdir(folder)
        torch.save(self.model.state_dict(), filepath)
