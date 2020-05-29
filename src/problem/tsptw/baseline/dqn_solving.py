import sys
import os
import numpy as np
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

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_arguments()

    sys.stdout.flush()
    rl_algorithm = "dqn"

    load_folder = "./selected-models/dqn/tsptw/n-city-%d/grid-%d-tw-%d-%d" % \
                  (args.n_city, args.grid_size, args.max_tw_gap, args.max_tw_size)

    solver_binding = SolverBinding(load_folder, args.n_city, args.grid_size, args.max_tw_gap,
                                   args.max_tw_size, args.seed, rl_algorithm)


    env = Environment(solver_binding.instance, solver_binding.n_node_feat, solver_binding.n_edge_feat,
                      1, args.grid_size, args.max_tw_gap, args.max_tw_size)

    cur_state = env.get_initial_environment()

    tour = [0]

    while True:

        graph = env.make_nn_input(cur_state, 'cpu')

        avail = env.get_valid_actions(cur_state)

        bgraph = dgl.batch([graph])

        available = avail.astype(bool)

        with torch.no_grad():
            res = solver_binding.model(bgraph, graph_pooling=False)

        res = dgl.unbatch(res)

        out = [r.ndata["n_feat"].data.cpu().numpy().flatten() for r in res]
        out = out[0].reshape(-1)

        action_idx = np.argmax(out[available])


        action = np.arange(len(out))[available][action_idx]

        cur_state, reward = env.get_next_state_with_reward(cur_state, action)
        tour.append(action)
        if cur_state.is_done():
            tour.append(0)
            break

    travel_time = 0

    if len(tour) != args.n_city + 1:
        print("TOUR:", tour)
        print("NO COMPLETE TOUR FOUND")

    else:
        for i in range(args.n_city):
            travel_time += solver_binding.instance.travel_time[tour[i]][tour[i + 1]]
        print("TOUR:", tour)
        print("BEST SOLUTION:", travel_time)