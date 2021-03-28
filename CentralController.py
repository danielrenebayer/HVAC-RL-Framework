
import os
import torch
import pickle
import datetime
import numpy as np
from copy import deepcopy

from ReplayBuffer import ReplayBufferStd
import StateUtilities as SU

from Agents import agent_constructor
import RLCritics

def ddpg_episode_mc(building, building_occ, agents, critics, output_lists,
        hyper_params = None, episode_number = 0, sqloutput = None,
        extended_logging = False, evaluation_epoch = False,
        add_ou_in_eval_epoch = False):
    #
    # define the hyper-parameters
    if hyper_params is None:
        LAMBDA_REWARD_ENERGY = 0.1
        LAMBDA_REWARD_MANU_STP_CHANGES = 150
        TAU_TARGET_NETWORKS = 0.01
        DISCOUNT_FACTOR = 0.9
        BATCH_SIZE = 128
        RPB_BUFFER_SIZE = 12*24*2 # 2 Tage
        LEARNING_RATE = 0.01
    else:
        LAMBDA_REWARD_ENERGY = hyper_params.lambda_rwd_energy
        LAMBDA_REWARD_MANU_STP_CHANGES = hyper_params.lambda_rwd_mstpc
        TAU_TARGET_NETWORKS  = hyper_params.tau
        DISCOUNT_FACTOR = hyper_params.discount_factor
        BATCH_SIZE      = hyper_params.batch_size
        RPB_BUFFER_SIZE = hyper_params.rpb_buffer_size
        LEARNING_RATE   = hyper_params.lr
    #
    # Define the replay ReplayBuffer
    rpb = ReplayBufferStd(size=RPB_BUFFER_SIZE, number_agents=len(agents))
    #
    # prepare the simulation
    state = building.model.reset()
    SU.fix_year_confussion(state)
    norm_state_ten = SU.unnormalized_state_to_tensor(state, building)
    #
    current_occupancy = building_occ.draw_sample( state["time"] )
    timestep   = 0
    last_state = None
    # start the simulation loop
    while not building.model.is_terminate():
        actions = list()
        
        currdate = state['time']
        #
        # request occupancy for the next state
        nextdate = state['time'] + datetime.timedelta(minutes=5)
        next_occupancy = building_occ.draw_sample(nextdate)
        #
        # propagate occupancy values to COBS / EnergyPlus
        for zonename, occd in next_occupancy.items():
            actions.append({"priority":        0,
                            "component_type": "Schedule:Constant",
                            "control_type":   "Schedule Value",
                            "actuator_key":  f"OCC-SCHEDULE-{zonename}",
                            "value":           next_occupancy[zonename]["relative number occupants"],
                            "start_time":      state['timestep'] + 1})

        #
        # request new actions from all agents
        agent_actions_dict = {}
        agent_actions_list = []
        for agent in agents:
            add_ou_process = True
            if evaluation_epoch and not add_ou_in_eval_epoch:
                add_ou_process = False
            new_action = agent.step_tensor(norm_state_ten,
                                           use_actor = True,
                                           add_ou    = add_ou_process)
            agent_actions_list.append( new_action )
            new_action_dict = agent.output_tensor_to_action_dict(new_action)
            agent_actions_dict[agent.name] = SU.backtransform_variables_in_dict(new_action_dict, inplace=True)

        #
        # send agent actions to the building object and obtaion the actions for COBS/eplus
        actions.extend( building.obtain_cobs_actions( agent_actions_dict, state["timestep"]+1 ) )

        #
        # send actions to EnergyPlus and obtian the new state
        norm_state_ten_last = norm_state_ten
        last_state = state
        timestep  += 1
        state      = building.model.step(actions)
        current_occupancy = next_occupancy
        SU.fix_year_confussion(state)

        current_energy_Wh = state["energy"] / 360

        #
        # modify state
        norm_state_ten = SU.unnormalized_state_to_tensor(state, building)

        #
        # send current temp/humidity values for all rooms
        # obtain number of manual setpoint changes
        _, n_manual_stp_changes = building_occ.manual_setpoint_changes(state['time'], state["temperature"], None)

        #
        # reward computation
        if not hyper_params is None and hyper_params.alternate_reward:
            reward = - LAMBDA_REWARD_MANU_STP_CHANGES * alternate_reward_fn(state, building)
        else:
            reward = -( LAMBDA_REWARD_ENERGY * current_energy_Wh + LAMBDA_REWARD_MANU_STP_CHANGES * n_manual_stp_changes )

        #
        # save (last_state, actions, reward, state) to replay buffer
        rpb.add_transition(norm_state_ten_last, agent_actions_list, reward, norm_state_ten)

        #
        # sample minibatch
        b_state1, b_action, b_action_cat, b_reward, b_state2 = rpb.sample_minibatch(BATCH_SIZE)

        #
        # loop over all [agent, critic]-pairs
        output_loss_list = []
        output_q_st2_list= []
        output_J_mean_list=[]
        output_cr_frobnorm_mat_list = []
        output_cr_frobnorm_bia_list = []
        output_ag_frobnorm_mat_list = []
        output_ag_frobnorm_bia_list = []
        for agent, critic in zip(agents, critics):
            #
            # compute y
            #  Hint: s_{i+1} <- state2; s_i <- state1
            critic.model.zero_grad()
            #  1. compute mu'(s_{i+1})
            mu_list = [ aInnerLoop.step_tensor(b_state2, use_actor = False) for aInnerLoop in agents ]
            #  2. compute y
            q_st2 = critic.forward_tensor(b_state2, mu_list, no_target = False)
            y     = b_reward.detach() + DISCOUNT_FACTOR * q_st2
            # compute Q for state1
            q = critic.forward_tensor(b_state1, b_action_cat, no_target = True)
            # update critic by minimizing the loss L
            L = critic.compute_loss_and_optimize(q, y, no_backprop = evaluation_epoch)
            #
            # update actor policies
            # policy loss = J
            mu_list = [ aInnerLoop.step_tensor(b_state1, add_ou = False) for aInnerLoop in agents ]
            agent.model_actor.zero_grad()
            policy_J = -critic.forward_tensor(b_state1, mu_list)
            policy_J_mean = policy_J.mean()
            if not evaluation_epoch:
                policy_J_mean.backward()
                agent.optimizer_step()
            #
            # save outputs
            output_loss_list.append(float(L.detach().numpy()))
            output_q_st2_list.append(float(q_st2.detach().mean().numpy()))
            output_J_mean_list.append(float(policy_J_mean.detach().numpy()))
            # compute and store frobenius norms for the weights
            cr_fnorm1, cr_fnorm2, ag_fnorm1, ag_fnorm2 = 0, 0, 0, 0
            for p in critic.model.parameters():
                if len(p.shape) == 1: cr_fnorm2 += float(p.cpu().norm().detach().cpu().numpy())
                else: cr_fnorm1 += float(p.norm().detach().cpu().numpy())
            for p in agent.model_actor.parameters():
                if len(p.shape) == 1: ag_fnorm2 += float(p.norm().detach().cpu().numpy())
                else: ag_fnorm1 += float(p.norm().detach().cpu().numpy())
            output_cr_frobnorm_mat_list.append( cr_fnorm1 )
            output_cr_frobnorm_bia_list.append( cr_fnorm2 )
            output_ag_frobnorm_mat_list.append( ag_fnorm1 )
            output_ag_frobnorm_bia_list.append( ag_fnorm2 )


        if not evaluation_epoch:
            #
            # update target critic
            for critic in critics:
                critic.update_target_network(TAU_TARGET_NETWORKS)

            #
            # update target network for actor
            for agent in agents:
                agent.update_target_network(TAU_TARGET_NETWORKS)

        #
        # store losses in the loss list
        if not sqloutput is None:
            sqloutput.add_every_step_of_episode( locals() )

        #
        # store detailed output, if extended logging is selected
        if extended_logging and not sqloutput is None:
            sqloutput.add_every_step_of_some_episodes( locals() )

        if timestep % 20 == 0:
            print(f"episode {episode_number:3}, timestep {timestep:5}: {state['time']}")

    #
    # elements, that should be stored only once per episode
    if not sqloutput is None:
        sqloutput.add_last_step_of_episode( locals() )





def one_baseline_episode(building, building_occ, args, sqloutput = None):
    hyper_params = args
    episode_number = 0
    #
    # define the hyper-parameters
    if hyper_params is None:
        LAMBDA_REWARD_ENERGY = 0.1
        LAMBDA_REWARD_MANU_STP_CHANGES = 150
        TAU_TARGET_NETWORKS = 0.01
        DISCOUNT_FACTOR = 0.9
        BATCH_SIZE = 128
        RPB_BUFFER_SIZE = 12*24*2 # 2 Tage
        LEARNING_RATE = 0.01
    else:
        LAMBDA_REWARD_ENERGY = hyper_params.lambda_rwd_energy
        LAMBDA_REWARD_MANU_STP_CHANGES = hyper_params.lambda_rwd_mstpc
        TAU_TARGET_NETWORKS  = hyper_params.tau
        DISCOUNT_FACTOR = hyper_params.discount_factor
        BATCH_SIZE      = hyper_params.batch_size
        RPB_BUFFER_SIZE = hyper_params.rpb_buffer_size
        LEARNING_RATE   = hyper_params.lr
    #
    # Set model parameters
    episode_len        = args.episode_length
    episode_start_day  = args.episode_start_day
    episode_start_month= args.episode_start_month
    building.model.set_runperiod(episode_len, 2017, episode_start_month, episode_start_day)
    building.model.set_timestep(12) # 5 Min interval, 12 steps per hour
    #
    # Define the agents
    agents = []
    for agent_name, (controlled_device, controlled_device_type) in building.agent_device_pairing.items():
        new_agent = agent_constructor( controlled_device_type[:-2] + "NoRL" )
        new_agent.initialize(
                         name = agent_name,
                         controlled_element = controlled_device)
        agents.append(new_agent)
    #
    # prepare the simulation
    state = building.model.reset()
    SU.fix_year_confussion(state)
    #
    current_occupancy = building_occ.draw_sample( state["time"] )
    timestep   = 0
    last_state = None
    # start the simulation loop
    while not building.model.is_terminate():
        actions = list()
        
        currdate = state['time']
        #
        # request occupancy for the next state
        nextdate = state['time'] + datetime.timedelta(minutes=5)
        next_occupancy = building_occ.draw_sample(nextdate)
        #
        # propagate occupancy values to COBS / EnergyPlus
        for zonename, occd in next_occupancy.items():
            actions.append({"priority":        0,
                            "component_type": "Schedule:Constant",
                            "control_type":   "Schedule Value",
                            "actuator_key":  f"OCC-SCHEDULE-{zonename}",
                            "value":           next_occupancy[zonename]["relative number occupants"],
                            "start_time":      state['timestep'] + 1})

        #
        # request new actions from all agents
        agent_actions_dict = {}
        for agent in agents:
            agent_actions_dict[agent.name] = agent.step(state)

        #
        # send agent actions to the building object and obtaion the actions for COBS/eplus
        actions.extend( building.obtain_cobs_actions( agent_actions_dict, state["timestep"]+1 ) )

        #
        # send actions to EnergyPlus and obtian the new state
        last_state = state
        timestep  += 1
        state      = building.model.step(actions)
        current_occupancy = next_occupancy
        SU.fix_year_confussion(state)

        current_energy_Wh = state["energy"] / 360

        # modify state
        norm_state_ten = SU.unnormalized_state_to_tensor(state, building)

        #
        # send current temp/humidity values for all rooms
        # obtain number of manual setpoint changes
        _, n_manual_stp_changes = building_occ.manual_setpoint_changes(currdate, state["temperature"], None)

        #
        # reward computation
        reward = -( LAMBDA_REWARD_ENERGY * current_energy_Wh + LAMBDA_REWARD_MANU_STP_CHANGES * n_manual_stp_changes )

        #
        # store losses in the loss list
        if not sqloutput is None:
            sqloutput.add_every_step_of_episode( locals(), ignore_agents = True )

        #
        # store detailed output
        if not sqloutput is None:
            sqloutput.add_every_step_of_some_episodes( locals() )

        if timestep % 20 == 0:
            print(f"baseline episode, timestep {timestep:5}: {state['time']}")

    # commit sql output if available
    if not sqloutput is None: sqloutput.db.commit()






def run_for_n_episodes(n_episodes, building, building_occ, args, sqloutput = None):
    """
    Runs the ddpg algorithm (i.e. the above defined ddpg_episode_mc function)
    for n_episodes runs.
    The agents and critics will be initialized according to the building object.
    """

    #
    # Define the agents
    agents = []
    # HINT: a device can be a zone, too
    for agent_name, (controlled_device, controlled_device_type) in building.agent_device_pairing.items():
        new_agent = agent_constructor( controlled_device_type )
        new_agent.initialize(
                         name = agent_name,
                         args = args,
                         controlled_element = controlled_device,
                         global_state_keys  = building.global_state_variables)
        agents.append(new_agent)

    #
    # Define the critics
    critics = []
    ciritic_input_variables=["Minutes of Day","Day of Week","Calendar Week",
                             "Outdoor Air Temperature","Outdoor Air Humidity",
                             "Outdoor Wind Speed","Outdoor Wind Direction",
                             "Outdoor Solar Radi Diffuse","Outdoor Solar Radi Direct"]
    for vartype in ["Zone Temperature","Zone People Count",
                    "Zone Relative Humidity",
                    "Zone VAV Reheat Damper Position","Zone CO2"]:
        ciritic_input_variables.extend( [f"SPACE{k}-1 {vartype}" for k in range(1,6)] )
    for agent in agents:
        new_critic = RLCritics.CriticMergeAndOnlyFC(
                    args = args,
                    input_variables=ciritic_input_variables,
                    agents = agents,
                    global_state_keys=building.global_state_variables)
        critics.append(new_critic)

    #
    # Load existing models if selected
    if args.load_models_from_path != "":
        load_episode = args.load_models_episode
        load_path    = os.path.abspath(args.load_models_from_path)
        for idx, agent in enumerate(agents):
            agent.load_models_from_disk(load_path,  prefix=f"episode_{load_episode}_agent_{idx}")
            print(f"Agent {idx} loaded from {load_path}")
        for idx, critic in enumerate(critics):
            critic.load_models_from_disk(load_path, prefix=f"episode_{load_episode}_critic_{idx}")
            print(f"Critic {idx} loaded from {load_path}")

    #
    # Set model parameters
    episode_len        = args.episode_length
    episode_start_day  = args.episode_start_day
    episode_start_month= args.episode_start_month
    building.model.set_runperiod(episode_len, 2017, episode_start_month, episode_start_day)
    building.model.set_timestep(12) # 5 Min interval, 12 steps per hour

    for n_episode in range(n_episodes):
        output_lists = {

            "room_temp_list": [],
            "occupancy_list": [],
            "humidity_list": [],
            "co2_ppm_list": [],

            "vav_pos_list": []
        }
    
        ddpg_episode_mc(building,
                        building_occ,
                        agents,
                        critics,
                        output_lists,
                        args,
                        n_episode,
                        sqloutput,
                        (n_episode+1) % args.network_storage_frequency == 0,
                        (n_episode+1) % args.network_storage_frequency == 0,
                        args.add_ou_in_eval_epoch)

        # save agent/critic networks every selected run
        if (n_episode+1) % args.network_storage_frequency == 0:
            for idx, agent in enumerate(agents):
                agent.save_models_to_disk(args.checkpoint_dir, prefix=f"episode_{n_episode}_agent_{idx}")
            for idx, critic in enumerate(critics):
                critic.save_models_to_disk(args.checkpoint_dir, prefix=f"episode_{n_episode}_critic_{idx}")

        # commit sql output if available
        if not sqloutput is None: sqloutput.db.commit()





def alternate_reward_fn(state, building):
    changed_magnitude = 0
    dto = state['time']
    temp_values = state['temperature']
    for room in building.room_names:
        if dto.weekday() < 5 and dto.hour >= 7 and dto.hour < 18:
            # if the temperature is not in the range [21,23.5], change the setpoint
            if temp_values[room] < 21.0:
                changed_magnitude += 21.0 - temp_values[room]
            elif temp_values[room] > 23.5:
                changed_magnitude += temp_values[room] - 23.5
        else:
            # if the temperature is not in the range [15,17], change the setpoint
            if temp_values[room] < 15:
                changed_magnitude += 15 - temp_values[room]
            elif temp_values[room] > 17:
                changed_magnitude += temp_values[room] - 17
    return changed_magnitude






