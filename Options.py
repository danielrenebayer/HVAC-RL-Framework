
import argparse

def get_argparser():
    """
    Returns the default argparser for command-line arguments for the training scripts.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--tau', type=float, default=0.01)
    parser.add_argument('--discount_factor', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--rpb_buffer_size', type=int, default=12*24*2)
    parser.add_argument('--lambda_rwd_mstpc', type=float, default=150)
    parser.add_argument('--lambda_rwd_energy', type=float, default=0.1)
    parser.add_argument('--use_cuda', action='store_true')
    parser.add_argument('--ou.theta', type=float, default=0.3)
    parser.add_argument('--ou.mu', type=float, default=0.0)
    parser.add_argument('--ou.sigma', type=float, default=0.3)
    parser.add_argument('--episodes_count', type=int, default=100, help="Number of episodes to train on")
    parser.add_argument('--episode_length', type=int, default=30, help="The length of an episode in days")
    parser.add_argument('--episode_start_day', type=int, default=1)
    parser.add_argument('--episode_start_month', type=int, default=7)
    parser.add_argument('--critic_hidden_size', type=int, default=40)
    parser.add_argument('--number_occupants', type=int, default=40)

    return parser


