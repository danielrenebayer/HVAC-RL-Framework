
import argparse
import datetime

def get_argparser():
    """
    Returns the default argparser for command-line arguments for the training scripts.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default="ddpg", choices=["ddpg", "ddqn", "baseline_rule-based"], help="If 'baseline_rule-based' is selected, it will perform a evaluation for one episode only, instead of a complete training with a RL agent and critics.")
    parser.add_argument('--ddqn_new', action='store_true', help="Use DDQN implementation from as proposed by H.v.Hasselt")
    parser.add_argument('--ts_per_hour', type=int, default=12, help="Numer of timesteps per hour. Should be a divisor of 60, to get a perfect matching minutes resolution.")
    parser.add_argument('--ts_until_regulation', type=int, default=1, help="Number of timesteps until changes are forwarded to EPlus.")
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--tau', type=float, default=0.01)
    parser.add_argument('--discount_factor', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--rpb_buffer_size', type=int, default=12*24*2)
    parser.add_argument('--lambda_rwd_mstpc', type=float, default=0.2)
    parser.add_argument('--lambda_rwd_energy', type=float, default=0.001)
    parser.add_argument('--reward_function', type=str, default="sum_energy_mstpc", choices=[
        "sum_energy_mstpc", "rulebased_roomtemp",
        "sum_emean_ediff_mstpc",
        "rulebased_agent_output"
        ], help="Select the reward function. Alternate reward: ignore energy and manual setpoint changes, use rulebased reward instead.")
    parser.add_argument('--energy_cons_in_kWh', action='store_true', help="Change loss computation to use energy consumption in kWh, not Wh.")
    parser.add_argument('--log_reward', action='store_true', help="If set, the logarithm to base e is applied to -reward+1.")
    parser.add_argument('--log_rwd_energy', action='store_true', help="Do not use the total energy consumption, apply logarithm to it.")
    parser.add_argument('--reward_scale', type=float, default=1.0, help="Scaling factor for the final reward. Must not be negative.")
    parser.add_argument('--reward_offset', type=float, default=0.0, help="Offset to add on the final reward. Can be negative too.")
    parser.add_argument('--stp_reward_function', type=str, default="linear", choices=["linear", "quadratic", "cubic", "exponential"], help="Function to apply to number of setpoint changes / rulebased missmatches / agent-output missmatches, but NOT on the energy reward")
    parser.add_argument('--stp_reward_step_offset', type=float, default=0.0, help="Offset for manual setpoint changes.")
    parser.add_argument('--clip_econs_at', type=float, default=0.0, help="If set to a value > 0, it will clip the energy consumption at this level.")
    parser.add_argument('--soften_instead_of_clipping', action='store_true', help="If --clip_econs_at is set, to not hard clip at that level, multiply values high than the clip value with 0.1")
    parser.add_argument('--use_cuda', action='store_true')
    parser.add_argument('--ddqn_loss', type=str, default="L2", choices=["L1","L2"])
    parser.add_argument('--add_ou_in_eval_epoch', action='store_true', help="Adds the ou process also during evaluation epochs")
    parser.add_argument('--target_network_update_freq', type=int, default=3)
    parser.add_argument('--ou_theta', type=float, default=0.3)
    parser.add_argument('--ou_mu', type=float, default=0.0)
    parser.add_argument('--ou_sigma', type=float, default=0.3)
    parser.add_argument('--ou_update_freq', type=int, default=1, help="Number of steps until obtaining the next sample from the OU-process")
    parser.add_argument('--epsilon_initial', type=float, default=1.0, help="The initial value of epsilon.")
    parser.add_argument('--epsilon', type=float, default=0.05, help="The final epsilon value for random sampling in DDQN learning.")
    parser.add_argument('--epsilon_final_step', type=int, default=100, help="The timestep at wich the minimal epsilon value (as defined in epsilon-parameter) should be reached, only for DDQN learning.")
    parser.add_argument('--epsilon_decay_mode', type=str, default="exponential", choices=["exponential", "linear"])
    parser.add_argument('--episodes_count', type=int, default=100, help="Number of episodes to train on")
    parser.add_argument('--episode_length', type=int, default=30, help="The length of an episode in days")
    parser.add_argument('--episode_start_day', type=int, default=1)
    parser.add_argument('--episode_start_month', type=int, default=7)
    parser.add_argument('--critic_network', type=str, default="2HiddenLayer,FastPyramid")
    parser.add_argument('--agent_network', type=str, default="2HiddenLayer,Trapezium")
    parser.add_argument('--agent_init_fn', type=str, default="xavier_normal", choices=["xavier_normal", "he_normal", "normal"])
    parser.add_argument('--agent_init_gain', type=float, default=0.8, help="Gain for Xavier normal initialization")
    parser.add_argument('--agent_init_mean', type=float, default=0.0, help="Mean for normal initialization")
    parser.add_argument('--agent_init_std', type=float, default=0.25, help="Std. deviation for normal initialization")
    parser.add_argument('--use_layer_normalization', action='store_true', help="Add layer normalization layers bevore every actiavtion function")
    parser.add_argument('--fewer_q_values', action='store_true', help="Reduce the number of actions an agent can take.")
    parser.add_argument('--agent_w_l2', type=float, default=0.00001, help="L2 penalty for agent parameters")
    parser.add_argument('--critic_w_l2', type=float, default=0.00001, help="L2 penalty for critic parameters")
    parser.add_argument('--optimizer', type=str, default="adam", choices=["adam", "sgd", "rmsprop"])
    parser.add_argument('--network_storage_frequency',type=int, default=10, help="Number of episodes until the next storage of the networks (critcs and agents)")
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints/" + datetime.datetime.now().strftime("%Y%m%d-%H%M"))
    parser.add_argument('--shared_network_per_agent_class', action='store_true')
    parser.add_argument('--model', type=str, default="Building_5ZoneAirCooled", choices=[
        "Building_5ZoneAirCooled", "Building_5ZoneAirCooled_SmallAgents",
        "Building_5ZoneAirCooled_SingleSetpoint", "Building_5ZoneAirCooled_SingleSetpoint_SmallAgents",
        "Building_5ZoneAirCooled_SingleSetpoint_SingleBIGAgent",
        "Building_5ZoneAirCooled_SingleSetpoint_SingleAgent",
        "Building_5ZoneAirCooled_SingleAgent", "Building_5ZoneAirCooled_SmallSingleAgent"])
    parser.add_argument('--single_setpoint_agent_count', type=str, default="all", choices=["all", "one", "two", "three", "one_but3not5", "one_but2not5", "two_but24not35", "three_but124not345"])
    parser.add_argument('--number_occupants', type=int, default=35)
    parser.add_argument('--next_occ_horizont', type=int, default=0, help="Numer of future occupancy states that are added to the state dict.")
    parser.add_argument('--load_models_from_path', type=str, default="", help="Path to the pickle objects for the agent and critic network(s). Do not load something, if set to an empty string (default).")
    parser.add_argument('--load_models_episode', type=int, default=0, help="The episode to load. If load_models_from_path is set to an empty string, it will be ignored.")
    parser.add_argument('--continue_training', action='store_true', help="If this parameter is given, the controller loads the existing models from the checkpoints dir. If the file `status.pickle` does not exist in checkpoints dir, it will start a new training run.")
    parser.add_argument('--eplus_storage_mode', action='store_true', help="Disable the restarting of EnergyPlus simulation at the beginning of a episode, use the old values instead again")
    parser.add_argument('--idf_file', type=str, default="", help="Path to the EPlus IDF file.")
    parser.add_argument('--epw_file', type=str, default="", help="Path to the EPlus weather file.")
    parser.add_argument('--rulebase_with_VAV', action='store_true', help="Should the rulebased agent control VAV damper position?")
    parser.add_argument('--rulebased_setpoint_unoccu_mean',  type=float, default=22.0)
    parser.add_argument('--rulebased_setpoint_unoccu_delta', type=float, default= 7.0)
    parser.add_argument('--rulebased_setpoint_occu_mean',    type=float, default=22.0)
    parser.add_argument('--rulebased_setpoint_occu_delta',   type=float, default= 1.0)
    parser.add_argument('--output_Q_vals_iep',  action="store_true", help="Output Q values in evaluation episodes.")
    parser.add_argument('--verbose_output_mode', action="store_true")

    return parser


