import sys
import os
import argparse
import dgl
import torch

sys.path.append(os.path.join(sys.path[0],'..','..','..','..'))

from src.problem.tsptw.solving.solver_binding import SolverBinding
from src.problem.tsptw.environment.environment import Environment

def parse_arguments():
    parser = argparse.ArgumentParser()

    # Instances parameters
    parser.add_argument('--n_city', type=int, default=20)
    parser.add_argument('--grid_size', type=int, default=100)
    parser.add_argument('--max_tw_gap', type=int, default=10)
    parser.add_argument('--max_tw_size', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--beam_size', type=int, default=4)
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_arguments()

    sys.stdout.flush()
    rl_algorithm = "ppo"


    load_folder = "./selected-models/ppo/tsptw/n-city-%d/grid-%d-tw-%d-%d" % \
                  (args.n_city, args.grid_size, args.max_tw_gap, args.max_tw_size)

    solver_binding = SolverBinding(load_folder, args.n_city, args.grid_size, args.max_tw_gap,
                                   args.max_tw_size, args.seed, rl_algorithm)


    env = Environment(solver_binding.instance, solver_binding.n_node_feat, solver_binding.n_edge_feat,
                      1, args.grid_size, args.max_tw_gap, args.max_tw_size)

    cur_state = env.get_initial_environment()

    sequences = [[[0], cur_state, 1.0]]

    total_reward = 0

    for _ in range(args.n_city - 1):

        all_candidates = list()

        for i in range(len(sequences)):
            seq, state, score = sequences[i]

            if state is not None:
                graph = env.make_nn_input(state, 'cpu')
                avail = env.get_valid_actions(state)

                available_tensor = torch.FloatTensor(avail)

                bgraph = dgl.batch([graph, ])

                res = solver_binding.model(bgraph, graph_pooling=False)

                out = dgl.unbatch(res)[0]

                action_probs = out.ndata["n_feat"].squeeze(-1)

                action_probs = action_probs + torch.abs(torch.min(action_probs))
                action_probs = action_probs - torch.max(action_probs * available_tensor)

                action_probs = solver_binding.actor_critic_network.masked_softmax(action_probs, available_tensor, dim=0,
                                                                         temperature=1)
                action_probs = action_probs.detach()

            for j in range(args.n_city):

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

    best_travel_time = 10000000
    best_seq = []
    for k in range(len(bs_tour)):
        seq = bs_tour[k] + [0]
        travel_time_acc = 0
        for i in range(args.n_city):
            travel_time_acc += solver_binding.instance.travel_time[seq[i]][seq[i + 1]]

        if travel_time_acc < best_travel_time:
            best_seq = seq
            best_travel_time = travel_time_acc
    print("TOUR:", seq)
    print("BEST SOLUTION:", best_travel_time)