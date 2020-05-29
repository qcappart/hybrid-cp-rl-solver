
import sys
import os
import argparse

sys.path.append(os.path.join(sys.path[0],'..','..','..'))

from src.problem.tsptw.learning.trainer_ppo import TrainerPPO



os.environ['KMP_DUPLICATE_LIB_OK']='True'


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Instances parameters
    parser.add_argument('--n_city', type=int, default=20)
    parser.add_argument('--grid_size', type=int, default=100)
    parser.add_argument('--max_tw_gap', type=int, default=10)
    parser.add_argument('--max_tw_size', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1)


    # Hyper parameters
    parser.add_argument('--n_episode', type=int, default=1000000)
    parser.add_argument('--learning_rate', type=float, default=0.002)
    parser.add_argument('--update_timestep', type=int, default=2000)
    parser.add_argument('--eps_clip', type=float, default=0.2)
    parser.add_argument('--entropy_value', type=float, default=0.01)
    parser.add_argument('--k_epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--hidden_layer', type=int, default=3)

    # Argument for Trainer
    parser.add_argument('--save_dir', type=str, default='./result-default')
    parser.add_argument('--plot_training', type=int, default=1)
    parser.add_argument('--mode', default='cpu', help='cpu/gpu')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_arguments()

    print("***********************************************************")
    print("[INFO] TRAINING ON RANDOM INSTANCES: TSPTW")
    print("[INFO] n_city: %d" % args.n_city)
    print("[INFO] grid_size: %d" % args.grid_size)
    print("[INFO] max_tw_gap: %d" % args.max_tw_gap)
    print("[INFO] max_tw_size: %d" % args.max_tw_size)
    print("[INFO] seed: %s" % args.seed)
    print("***********************************************************")
    print("[INFO] TRAINING PARAMETERS")
    print("[INFO] Algorithm: PPO")
    print("[INFO] learning rate: %f" % args.learning_rate)
    print("[INFO] eps_clip: %f" % args.eps_clip)
    print("[INFO] entropy_value: %f" % args.entropy_value)
    print("[INFO] hidden_layer: %d" % args.hidden_layer)
    print("[INFO] k_epochs: %d" % args.k_epochs)
    print("[INFO] batch_size: %d" % args.batch_size)
    print("[INFO] update_timestep: %d" % args.update_timestep)
    print("[INFO] latent_dim: %d" % args.latent_dim)
    print("***********************************************************")
    sys.stdout.flush()

    trainer = TrainerPPO(args)
    trainer.run_training()
