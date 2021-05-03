
import argparse
import datetime

def get_argparser():
    """
    Returns the default argparser for command-line arguments for the training scripts.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default="ddpg", choices=["ddpg", "ddqn", "baseline_rule-based"], help="If 'baseline_rule-based' is selected, it will perform a evaluation for one episode only, instead of a complete training with a RL agent and critics.")
    parser.add_argument('--ts_per_hour', type=int, default=12, help="Numer of timesteps per hour. Should be a divisor of 60, to get a perfect matching minutes resolution.")
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--tau', type=float, default=0.01)
    parser.add_argument('--discount_factor', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--rpb_buffer_size', type=int, default=12*24*2)
    parser.add_argument('--lambda_rwd_mstpc', type=float, default=0.2)
    parser.add_argument('--lambda_rwd_energy', type=float, default=0.001)
    parser.add_argument('--reward_function', type=str, default="sum_energy_mstpc", choices=[
        "sum_energy_mstpc", "rulebased_roomtemp",
        "rulebased_agent_output"
        ], help="Select the reward function. Alternate reward: ignore energy and manual setpoint changes, use rulebased reward instead.")
    parser.add_argument('--log_reward', action='store_true', help="If set, the logarithm to base e is applied to -reward+1.")
    parser.add_argument('--use_cuda', action='store_true')
    parser.add_argument('--add_ou_in_eval_epoch', action='store_true', help="Adds the ou process also during evaluation epochs")
    parser.add_argument('--target_network_update_freq', type=int, default=3)
    parser.add_argument('--ou_theta', type=float, default=0.3)
    parser.add_argument('--ou_mu', type=float, default=0.0)
    parser.add_argument('--ou_sigma', type=float, default=0.3)
    parser.add_argument('--ou_update_freq', type=int, default=1, help="Number of steps until obtaining the next sample from the OU-process")
    parser.add_argument('--epsilon', type=float, default=0.05, help="The final epsilon value for random sampling in DDQN learning.")
    parser.add_argument('--epsilon_final_step', type=int, default=100, help="The timestep at wich the minimal epsilon value (as defined in epsilon-parameter) should be reached, only for DDQN learning.")
    parser.add_argument('--episodes_count', type=int, default=100, help="Number of episodes to train on")
    parser.add_argument('--episode_length', type=int, default=30, help="The length of an episode in days")
    parser.add_argument('--episode_start_day', type=int, default=1)
    parser.add_argument('--episode_start_month', type=int, default=7)
    parser.add_argument('--critic_hidden_size', type=int, default=40)
    parser.add_argument('--critic_hidden_activation', type=str, default="tanh", choices=["tanh","LeakyReLU"])
    parser.add_argument('--critic_last_activation',   type=str, default="tanh", choices=["tanh","LeakyReLU"])
    parser.add_argument('--agent_w_l2', type=float, default=0.00001, help="L2 penalty for agent parameters")
    parser.add_argument('--critic_w_l2', type=float, default=0.00001, help="L2 penalty for critic parameters")
    parser.add_argument('--optimizer', type=str, default="adam", choices=["adam", "sgd", "rmsprop"])
    parser.add_argument('--network_storage_frequency',type=int, default=10, help="Number of episodes until the next storage of the networks (critcs and agents)")
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints/" + datetime.datetime.now().strftime("%Y%m%d-%H%M"))
    parser.add_argument('--shared_network_per_agent_class', action='store_true')
    parser.add_argument('--model', type=str, default="Building_5ZoneAirCooled", choices=[
        "Building_5ZoneAirCooled", "Building_5ZoneAirCooled_SmallAgents",
        "Building_5ZoneAirCooled_SingleSetpoint", "Building_5ZoneAirCooled_SingleSetpoint_SmallAgents",
        "Building_5ZoneAirCooled_SingleSetpoint_SingleSmallAgent",
        "Building_5ZoneAirCooled_SingleSetpoint_SingleAgent",
        "Building_5ZoneAirCooled_SingleAgent", "Building_5ZoneAirCooled_SmallSingleAgent"])
    parser.add_argument('--number_occupants', type=int, default=40)
    parser.add_argument('--load_models_from_path', type=str, default="", help="Path to the pickle objects for the agent and critic network(s). Do not load something, if set to an empty string (default).")
    parser.add_argument('--load_models_episode', type=int, default=0, help="The episode to load. If load_models_from_path is set to an empty string, it will be ignored.")
    parser.add_argument('--continue_training', action='store_true', help="If this parameter is given, the controller loads the existing models from the checkpoints dir. If the file `status.pickle` does not exist in checkpoints dir, it will start a new training run.")
    parser.add_argument('--eplus_storage_mode', action='store_true', help="Disable the restarting of EnergyPlus simulation at the beginning of a episode, use the old values instead again")
    parser.add_argument('--idf_file', type=str, default="", help="Path to the EPlus IDF file.")
    parser.add_argument('--epw_file', type=str, default="", help="Path to the EPlus weather file.")
    parser.add_argument('--rulebased_setpoint_unoccu_mean',  type=float, default=23.0)
    parser.add_argument('--rulebased_setpoint_unoccu_delta', type=float, default= 7.0)
    parser.add_argument('--rulebased_setpoint_occu_mean',    type=float, default=21.5)
    parser.add_argument('--rulebased_setpoint_occu_delta',   type=float, default= 1.0)

    return parser


