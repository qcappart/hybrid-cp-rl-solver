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
    parser.add_argument('--beam_size', type=int, default=4)
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_arguments()

    sys.stdout.flush()
    rl_algorithm = "ppo"

    load_folder = "./selected-models/ppo/portfolio/n-item-%d/capacity-ratio-%.1f/moment-factors-%d-%d-%d-%d" % \
                  (args.n_item, args.capacity_ratio, args.lambda_1, args.lambda_2, args.lambda_3, args.lambda_4)

    solver_binding = SolverBinding(load_folder, args.n_item, args.capacity_ratio,
                                   args.lambda_1, args.lambda_2, args.lambda_3, args.lambda_4,
                                   args.discrete_coeff, args.seed, rl_algorithm)


    env = Environment(solver_binding.instance, solver_binding.n_feat, 1)

    cur_state = env.get_initial_environment()

    sequences = [[list(), cur_state, 1.0]]

    for _ in range(args.n_item):

        all_candidates = list()
        for i in range(len(sequences)):

            seq, state, score = sequences[i]

            if state is not None:
                state_feats = env.make_nn_input(state, "cpu")
                avail = env.get_valid_actions(state)
                available_tensor = torch.FloatTensor(avail)
                with torch.no_grad():
                    batched_set = state_feats.unsqueeze(0)
                    out = solver_binding.model(batched_set)
                    action_probs = out.squeeze(0)

                action_probs = action_probs + torch.abs(torch.min(action_probs))
                action_probs = action_probs - torch.max(action_probs * available_tensor)
                action_probs = solver_binding.actor_critic_network.masked_softmax(action_probs, available_tensor, dim=0,
                                                                         temperature=1)
                action_probs = action_probs.detach()

            for j in range(2):

                if state is not None and action_probs[j] > 10e-30:
                    next_state, reward = env.get_next_state_with_reward(state, j)
                    candidate = [seq + [j], next_state, score * action_probs[j].detach()]
                    all_candidates.append(candidate)
                else:
                    candidate = [seq + [-1], None, 0]
                    all_candidates.append(candidate)
            # order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup: -tup[2])
        # select k best
        sequences = ordered[:args.beam_size]

    bs_tour = [x for (x, y, z) in sequences]
    bs_tour = filter(lambda x: not -1 in x, bs_tour)
    bs_tour = list(bs_tour)
    best_profit = -1000000
    best_sol = []
    for k in range(len(bs_tour)):
        # print(bs_tour[k])

        if not args.discrete_coeff:
            tot_mean = sum([a * b for a, b in zip(bs_tour[k], env.instance.means)])
            tot_deviation = sum([a * b for a, b in zip(bs_tour[k], env.instance.deviations)]) ** (1. / 2)
            tot_skewness = sum([a * b for a, b in zip(bs_tour[k], env.instance.skewnesses)]) ** (1. / 3)
            tot_kurtosis = sum([a * b for a, b in zip(bs_tour[k], env.instance.kurtosis)]) ** (1. / 4)

        else:
            tot_mean = int(sum([a * b for a, b in zip(bs_tour[k], env.instance.means)]))
            tot_deviation = int(sum([a * b for a, b in zip(bs_tour[k], env.instance.deviations)]) ** (1. / 2))
            tot_skewness = int(sum([a * b for a, b in zip(bs_tour[k], env.instance.skewnesses)]) ** (1. / 3))
            tot_kurtosis = int(sum([a * b for a, b in zip(bs_tour[k], env.instance.kurtosis)]) ** (1. / 4))

        tot_profit = args.lambda_1 * tot_mean - \
                     args.lambda_2 * tot_deviation + \
                     args.lambda_3 * tot_skewness - \
                     args.lambda_4 * tot_kurtosis

        if tot_profit > best_profit:
            best_sol = bs_tour[k]
            best_profit = tot_profit

    print("ITEMS INSERTED:", best_sol)
    print("BEST SOLUTION:", best_profit)