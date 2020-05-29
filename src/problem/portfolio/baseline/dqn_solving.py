import sys
import os
import numpy as np
import argparse
import torch

sys.path.append(os.path.join(sys.path[0],'..','..','..','..'))

from src.problem.portfolio.solving.solver_binding import SolverBinding
from src.problem.portfolio.environment.environment import Environment

def parse_arguments():
    parser = argparse.ArgumentParser()

    # Instances parameters
    parser.add_argument('--n_item', type=int, default=10)
    parser.add_argument('--capacity_ratio', type=float, default=0.5)
    parser.add_argument('--lambda_1', type=int, default=1)
    parser.add_argument('--lambda_2', type=int, default=5)
    parser.add_argument('--lambda_3', type=int, default=5)
    parser.add_argument('--lambda_4', type=int, default=5)
    parser.add_argument('--discrete_coeff', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1)

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_arguments()

    sys.stdout.flush()
    rl_algorithm = "dqn"

    load_folder = "./selected-models/dqn/portfolio/n-item-%d/capacity-ratio-%.1f/moment-factors-%d-%d-%d-%d" % \
                  (args.n_item, args.capacity_ratio, args.lambda_1, args.lambda_2, args.lambda_3, args.lambda_4)

    solver_binding = SolverBinding(load_folder, args.n_item, args.capacity_ratio,
                                   args.lambda_1, args.lambda_2, args.lambda_3, args.lambda_4,
                                   args.discrete_coeff, args.seed, rl_algorithm)


    env = Environment(solver_binding.instance, solver_binding.n_feat, 1)

    cur_state = env.get_initial_environment()

    solution = []
    total_profit = 0

    while True:

        nn_input = env.make_nn_input(cur_state, 'cpu')

        avail = env.get_valid_actions(cur_state)

        nn_input = nn_input.unsqueeze(0)
        available = avail.astype(bool)

        with torch.no_grad():
            res = solver_binding.model(nn_input)

        out = res.cpu().numpy().squeeze(0)

        action_idx = np.argmax(out[available])

        action = np.arange(len(out))[available][action_idx]

        cur_state, reward = env.get_next_state_with_reward(cur_state, action)
        solution.append(action)
        total_profit += reward
        if cur_state.is_done():
            break

    print("ITEMS INSERTED:", solution)
    print("BEST SOLUTION:", total_profit)