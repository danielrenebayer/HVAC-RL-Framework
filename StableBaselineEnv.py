
import gym
from gym import spaces

import numpy as np
import datetime
from copy import deepcopy

import StateUtilities as SU
from Agents import agent_constructor
from CentralController import reward_fn_rulebased_roomtemp, reward_fn_rulebased_agent_output

class StableBaselineEnv(gym.Env):
    """
    Custom environment for accessing EPlus trough COBS, following the gym interface.
    Created by using infromations taken from https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
    """

    metadata = {'render.modes': ['human']}

    #
    # this class is equivalent to a single agent run using
    # "SingleSetpoint,SingleAgent,Q,RL"-agents and the
    # Building_5ZoneAirCooled_SingleSetpoint_SingleAgent building
    #

    def __init__(self, building, building_occ, args):
        super(StableBaselineEnv, self).__init__()
        #
        # Initialize the object and EPlus/COBS
        self.building = building
        self.building_occ = building_occ
        self.initialize_eplus_and_agents(args)
        #
        self.observation_list = [
            "Minutes of Day",
            "Day of Week",
            "Calendar Week",
            "Outdoor Air Temperature",
            'Outdoor Solar Radi Direct',
            "SPACE5-1 Zone People Count"]
        self.agent_action_name = "Zone Heating Setpoint"
        self.agent_actions     = np.linspace(14.0, 23.0, 10)
        #
        self.reward_function = args.reward_function
        self.LAMBDA_REWARD_MANU_STP_CHANGES = args.lambda_rwd_mstpc
        self.LAMBDA_REWARD_ENERGY           = args.lambda_rwd_energy
        self.reward_scale  = args.reward_scale
        self.reward_offset = args.reward_offset
        self.reward_list   = []
        self.reward_list_maxlen = 100
        self.ts_diff_in_min = 60 // args.ts_per_hour
        #
        # initialize the attributes required for gym env
        self.reward_range = (-float('inf'), 1.0)
        self.action_space = spaces.Discrete(len(self.agent_actions))
        self.observation_space = spaces.Box(low=-1, high=1, shape=(len(self.observation_list), ), dtype=np.float)

    def step(self, action):
        # action is an int
        #
        actions_for_cobs = list()
        currdate = self.state['time']
        # request occupancy for the next state
        nextdate = self.state['time'] + datetime.timedelta(minutes=self.ts_diff_in_min)
        next_occupancy = self.building_occ.draw_sample(nextdate)
        # propagate occupancy values to COBS / EnergyPlus
        for zonename, occd in next_occupancy.items():
            actions_for_cobs.append({
                            "priority":        0,
                            "component_type": "Schedule:Constant",
                            "control_type":   "Schedule Value",
                            "actuator_key":  f"OCC-SCHEDULE-{zonename}",
                            "value":           next_occupancy[zonename]["relative number occupants"],
                            "start_time":      self.state['timestep'] + 1})
        #
        #
        agent_actions_dict = {"MaiAgent": {"Zone Heating Setpoint": self.agent_actions[action]}}
        actions_for_cobs.extend( self.building.obtain_cobs_actions( agent_actions_dict, self.state["timestep"]+1 ) )
        #
        #
        self.state = self.building.model_step(actions_for_cobs)
        SU.fix_year_confussion(self.state)
        SU.expand_state_timeinfo_temp(self.state, self.building)
        self.state_normalized = SU.normalize_variables_in_dict(self.state)
        self.current_occupancy = next_occupancy
        #
        current_energy_Wh = self.state["energy"] / 360
        _, n_manual_stp_changes, target_temp_per_room = self.building_occ.manual_setpoint_changes(self.state['time'], self.state["temperature"], None)
        #
        # reward computation
        if self.reward_function == "sum_energy_mstpc":
            reward = self.LAMBDA_REWARD_ENERGY * current_energy_Wh + self.LAMBDA_REWARD_MANU_STP_CHANGES * n_manual_stp_changes
        elif self.reward_function == "rulebased_roomtemp":
            reward, target_temp_per_room = reward_fn_rulebased_roomtemp(self.state, self.building)
        else:
            reward, target_temp_per_room = reward_fn_rulebased_agent_output(self.state, agent_actions_dict, self.building)
        # invert and scale reward and (maybe) add offset
        reward = -self.reward_scale * reward + self.reward_offset
        self.add_to_reward_list(reward)
        done = self.building.model_is_terminate()
        #
        # output
        mean_rwd = np.mean(self.reward_list)
        print(f"Mean reward = {mean_rwd:10.7f} - action {action:2} taken - finished {done}")
        #
        return self.state_to_observation(), reward, done, {}

    def reset(self):
        self.state = self.building.model_reset()
        SU.fix_year_confussion(self.state)
        SU.expand_state_timeinfo_temp(self.state, self.building)
        self.state_normalized = SU.normalize_variables_in_dict(self.state)
        #
        self.current_occupancy = self.building_occ.draw_sample( self.state["time"] )
        self.last_state = None
        return self.state_to_observation()

    def render(self, mode='human'):
        print("Mean reward = ", np.mean(self.reward_list))

    def close(self):
        return

    def state_to_observation(self):
        """
        Converts the self.state object to a observation that can be passed to StabelBaseline
        """
        observation = []
        #print("normalized state: ", self.state_normalized)
        for keyname in self.observation_list:
            observation.append( self.state_normalized[keyname] )
        return  np.array(observation)

    def initialize_eplus_and_agents(self, args):
        #
        # Define the agents
        self.agents = []
        idx    = 0
        # HINT: a device can be a zone, too
        for agent_name, (controlled_device, controlled_device_type) in self.building.agent_device_pairing.items():
            new_agent = agent_constructor( controlled_device_type )
            new_agent.initialize(
                             name = agent_name,
                             args = args,
                             controlled_element = controlled_device,
                             global_state_keys  = self.building.global_state_variables)
            self.agents.append(new_agent)
            idx += 1
        #
        # Set model parameters
        episode_len        = args.episode_length
        episode_start_day  = args.episode_start_day
        episode_start_month= args.episode_start_month
        ts_diff_in_min     = 60 // args.ts_per_hour
        self.building.model.set_runperiod(episode_len, 2017, episode_start_month, episode_start_day)
        self.building.model.set_timestep( args.ts_per_hour )

    def add_to_reward_list(self, reward):
        if len(self.reward_list) >= self.reward_list_maxlen - 1:
            self.reward_list.pop()
        self.reward_list.append(reward)

