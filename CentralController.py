
import torch
import datetime
import numpy as np
from copy import deepcopy

from ReplayBuffer import ReplayBufferStd
import StateUtilities as SU

def ddpg_episode_mc(building, building_occ, agents, critics, output_lists, episode_number = 0, aux_output = {}):
    #
    # define the hyper-parameters
    LAMBDA_REWARD_ENERGY = 0.1
    LAMBDA_REWARD_MANU_STP_CHANGES = 150
    TAU_TARGET_NETWORKS = 0.01
    DISCOUNT_FACTOR = 0.9
    BATCH_SIZE = 128
    #RPB_BUFFER_SIZE = 12*24*30 # 30 Tage
    RPB_BUFFER_SIZE = 12*24*2 # 2 Tage
    LEARNING_RATE = 0.001
    #
    # Define the replay ReplayBuffer
    rpb = ReplayBufferStd(size=RPB_BUFFER_SIZE, number_agents=len(agents))
    #
    # prepare the simulation
    state = building.model.reset()
    norm_state_ten = SU.unnormalized_state_to_tensor(state, building)
    #
    #
    current_occupancy = building_occ.draw_sample(datetime.datetime(2020, 1, 1, 0, 0))
    timestep   = 0
    last_state = None
    # start the simulation loop
    while not building.model.is_terminate():
        actions = list()
        
        currdate = state['time']
        currdate = datetime.datetime(year=2020, month=currdate.month, day=currdate.day, hour=currdate.hour,
                                    minute=currdate.minute)
        
        #
        # request occupancy for the next state
        nextdate = state['time']
        nextdate = datetime.datetime(year=2020, month=nextdate.month, day=nextdate.day, hour=nextdate.hour,
                                    minute=nextdate.minute) + datetime.timedelta(minutes=5)
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
        # Read current status and save this to the dictionaries and lists
        # TODO: this is a function of the Building, should be located there
        output_lists["episode_list"].append(episode_number)
        output_lists["timestamp_list"].append(currdate)
        output_lists["room_temp_list"].append(state["temperature"])
        output_lists["occupancy_list"].append(current_occupancy)
        output_lists["outd_temp_list"].append(state["Outdoor Air Temperature"])
        output_lists["outd_humi_list"].append(state["Outdoor Air Humidity"])
        output_lists["outd_wdir_list"].append(state["Outdoor Wind Direction"])
        output_lists["outd_solar_radi_list"].append({
            "Direct Radiation": state["Outdoor Solar Radi Direct"],
            "Indirect Radiation": state["Outdoor Solar Radi Diffuse"]})
        output_lists["outd_wspeed_list"].append(state["Outdoor Wind Speed"])
        output_lists["energy_list"].append(   state["energy"])
        output_lists["co2_ppm_list"].append( {e: state[f"{e} Zone CO2"] for e in
                            [k.replace(' Zone CO2', "") for k in state.keys() if k.endswith(' Zone CO2')]} )
        output_lists["humidity_list"].append( {e: state[f"{e} Zone Relative Humidity"] for e in
                            [k.replace(' Zone Relative Humidity', "") for k in state.keys() if k.endswith(' Zone Relative Humidity')]} )
        #
        output_lists["vav_pos_list"].append( {e: state[f"{e} Zone VAV Reheat Damper Position"] for e in
                            [k.replace(' Zone VAV Reheat Damper Position', "") for k in state.keys() if k.endswith(' Zone VAV Reheat Damper Position')]} )
        


        #
        # request new actions from all agents
        agent_actions_dict = {}
        agent_actions_list = []
        for agent in agents:
            new_action = agent.step_tensor(norm_state_ten, use_actor = True, add_ou = True)
            agent_actions_list.append( new_action )
            agent_actions_dict[agent.name] = agent.output_tensor_to_action_dict(new_action)

        #
        # merge actions from all agents and convert them to actions from COBS/EPlus
        # TODO: this should be moved to the building
        for agent_name, ag_actions in agent_actions_dict.items():
            ag_actions = SU.backtransform_variables_in_dict(ag_actions)
            controlled_group, _ = building.agent_device_pairing[agent.name]
            if "VAV Reheat Damper Position" in ag_actions.keys():
                action_val = ag_actions["VAV Reheat Damper Position"]
                actions.append({"priority": 0,
                                "component_type": "Schedule:Constant",
                                "control_type": "Schedule Value",
                                "actuator_key": f"{controlled_group} VAV Customized Schedule",
                                "value": action_val,
                                "start_time": state['timestep'] + 1})
            if "Zone Heating/Cooling-Mean Setpoint"  in ag_actions.keys() and \
            "Zone Heating/Cooling-Delta Setpoint" in ag_actions.keys():
                mean_temp_sp = ag_actions["Zone Heating/Cooling-Mean Setpoint"]
                delta = ag_actions["Zone Heating/Cooling-Delta Setpoint"]
                if delta < 0.1: delta = 0.1
                actions.append({"value":      mean_temp_sp - delta,
                                "start_time": timestep + 1,
                                "priority":   0,
                                "component_type": "Zone Temperature Control",
                                "control_type":   "Heating Setpoint",
                                "actuator_key":   controlled_group})
                actions.append({"value":      mean_temp_sp + delta,
                                "start_time": timestep + 1,
                                "priority":   0,
                                "component_type": "Zone Temperature Control",
                                "control_type":   "Cooling Setpoint",
                                "actuator_key":   controlled_group})
            # ... das macht jetzt keinen Sinn mehr, eine Prüfung wäre aber vielleicht nicht schlecht
            #else:
            #    print(f"Action {action_name} from agent in zone {zone} unknown.")


        #
        # send actions to EnergyPlus and obtian the new state
        norm_state_ten_last = norm_state_ten
        last_state = state
        timestep  += 1
        state      = building.model.step(actions)
        current_occupancy = next_occupancy

        #
        # modify state
        norm_state_ten = SU.unnormalized_state_to_tensor(state, building)

        #
        # send current temp/humidity values for all rooms
        # obtain number of manual setpoint changes
        n_manual_stp_changes = building_occ.manual_setpoint_changes(currdate, state["temperature"], None)
        output_lists["n_manual_stp_ch_list"].append(n_manual_stp_changes)

        #
        # reward computation
        reward = -( LAMBDA_REWARD_ENERGY * state["energy"] + LAMBDA_REWARD_MANU_STP_CHANGES * n_manual_stp_changes )
        output_lists["rewards_list"].append(reward)

        #
        # save (last_state, actions, reward, state) to replay buffer
        rpb.add_transition(norm_state_ten_last, agent_actions_list, reward, norm_state_ten)
        aux_output["rpb"] = rpb

        #
        # sample minibatch
        b_state1, b_action, b_action_cat, b_reward, b_state2 = rpb.sample_minibatch(BATCH_SIZE)

        #
        # loop over all [agent, critic]-pairs
        output_loss_list = []
        for agent, critic in zip(agents, critics):
            #
            # compute y
            #  Hint: s_{i+1} <- state2; s_i <- state1
            critic.model.zero_grad()
            #  1. compute mu'(s_{i+1})
            aux_output["mu_list_input"] = b_state1
            mu_list = [ aInnerLoop.step_tensor(b_state2, use_actor = False) for aInnerLoop in agents ]
            aux_output["mu_list"] = mu_list
            #  2. compute y
            y = b_reward.detach() + DISCOUNT_FACTOR * critic.forward_tensor(b_state2, mu_list, no_target = False)
            # TODO: hier eventuell nochmal critic.model.zero_grad()
            # compute Q for state1
            q = critic.forward_tensor(b_state1, b_action_cat, no_target = True)
            # update critic by minimizing the loss L
            L = critic.compute_loss_and_optimize(q, y)
            output_loss_list.append(float(L.detach().numpy()))
            #
            # update actor policies
            # policy loss = J
            mu_list = [ aInnerLoop.step_tensor(b_state1, add_ou = False) for aInnerLoop in agents ]
            agent.model_actor.zero_grad()
            policy_J = -critic.forward_tensor(b_state1, mu_list)
            policy_J.mean().backward()
            agent.optimizer_step()

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
        output_lists["loss_list"].append(output_loss_list)

        if timestep % 20 == 0:
            print(f"episode {episode_number:3}, timestep {timestep:5}: {state['time']}")


