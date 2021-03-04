
import torch
import datetime
import numpy as np

from ReplayBuffer import ReplayBuffer
import StateUtilities as SU

def ddpg_episode(building, building_occ, agents, critic, output_lists, episode_number = 0):
    #
    # define the hyper-parameters
    LAMBDA_REWARD_ENERGY = 0.1
    LAMBDA_REWARD_MANU_STP_CHANGES = 150
    TAU_TARGET_NETWORKS = 0.01
    DISCOUNT_FACTOR = 0.9
    BATCH_SIZE = 100
    RPB_BUFFER_SIZE = 12*24*30 # 30 Tage
    LEARNING_RATE = 0.001
    #
    # Define the replay ReplayBuffer
    rpb = ReplayBuffer(RPB_BUFFER_SIZE)
    #
    # prepare the simulation
    state = building.model.reset()
    SU.expand_state_timeinfo_temp(state, building)
    normalized_state = SU.normalize_variables_in_dict(state)
    #
    # initialize the losses and optimizers
    critic_loss_mse = torch.nn.MSELoss()
    critic_optimizer = torch.optim.Adam(params = critic.model.parameters(), lr = LEARNING_RATE)
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
        agent_actions = {}
        agent_input_tensors = {}
        agent_output_tensors= {}
        for agent in agents:
            state_subset = SU.retrieve_substate_for_agent(normalized_state, agent, building)
            # send modified state and obtain new actions
            new_actions_ten, new_actions_dict, input_for_agent = agent.step(state_subset, add_ou=True)
            agent_actions[agent.name]        = new_actions_dict
            agent_output_tensors[agent.name] = new_actions_ten.detach()
            agent_input_tensors[agent.name]  = input_for_agent.detach()

        #
        # merge actions from all agents and convert them to actions from COBS/EPlus
        # TODO: this should be moved to the building
        for agent_name, ag_actions in agent_actions.items():
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
        normalized_last_state = normalized_state
        last_state = state
        timestep  += 1
        state      = building.model.step(actions)
        current_occupancy = next_occupancy

        #
        # modify state
        #   1. expand temperature and expand time info
        SU.expand_state_timeinfo_temp(state, building)
        #   2. normalize state
        normalized_state = SU.normalize_variables_in_dict(state)

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
        #   bevore we can do so, we have to compute the missing values, i.e.
        #     - "critic input state tensor"         for the last and the current state,
        #     - "critic merged agent action tensor" for the last state
        #     - "agents input state tensors"        for the current state
        agents_input_state_tensors_fcs = {agent.name: agent.prepare_state_dict(SU.retrieve_substate_for_agent(normalized_state, agent, building)) for agent in agents}
        critic_input_state_tensor_fcs = critic.prepare_state_dict(normalized_state)
        critic_input_state_tensor_fls = critic.prepare_state_dict(normalized_last_state)
        critic_merged_agend_action_ten= critic.prepare_action_dict(agent_actions)
        #   save to ReplayBuffer
        rpb.add_transition(
            {"normalized state": normalized_last_state,
             "agents input state tensors": agent_input_tensors,
             "critic input state tensor": critic_input_state_tensor_fls},
            {"agent output tensors": agent_output_tensors,
             "critic merged agent action tensor": critic_merged_agend_action_ten},
            reward,
            {"normalized state": normalized_state,
             "agents input state tensors": agents_input_state_tensors_fcs,
             "critic input state tensor": critic_input_state_tensor_fcs}
        )

        #
        # sample minibatch
        batch_state1, batch_action, batch_reward, batch_state2 = rpb.sample_minibatch(BATCH_SIZE)

        #
        # compute y and update critic
        #  s_{i+1} <- state2; s_i <- state1
        critic.model.zero_grad()
        #  compute mu'(s_{i+1})
        agent_outputs = []
        for agent in agents:
            agent_tensor_list = []
            for batch_element in batch_state2:
                agent_tensor_list.append(batch_element["agents input state tensors"][agent.name])
            #print(agent_tensor_list[0].shape)
            agent_tensor = torch.cat(agent_tensor_list)
            agent_outputs.append( agent.step_tensor(agent_tensor, use_actor=False) )
        agent_outputs_ten = torch.cat(agent_outputs, dim=1)
        #  merge vector for critic input state
        critic_input_state2_tensor = []
        for batch_element in batch_state2:
            critic_input_state2_tensor.append(batch_element["critic input state tensor"])
        critic_input_state2_tensor = torch.cat(critic_input_state2_tensor)
        y = batch_reward + DISCOUNT_FACTOR * critic.forward_tensor(critic_input_state2_tensor, agent_outputs_ten, no_target=False)
        #  compute Q(s_i, a_i)
        critic_input_state1_list = []
        for batch_element in batch_state1:
            critic_input_state1_list.append(batch_element["critic input state tensor"])
        critic_input_action_list = []
        for batch_element in batch_action:
            critic_input_action_list.append(batch_element["critic merged agent action tensor"])
        critic_input_state1_ten = torch.cat(critic_input_state1_list).detach()
        critic_input_action_ten = torch.cat(critic_input_action_list).detach()
        q_val = critic.forward_tensor(critic_input_state1_ten, critic_input_action_ten, no_target=True)
        #  update critic by maximizing L
        critic_loss_value = critic_loss_mse(y.detach(), q_val)
        critic_loss_value.backward()
        critic_optimizer.step()

        #
        # update actor policies
        agent_outputs = []
        for agent in agents:
            agent.model_actor.zero_grad()
            agent_input_list = []
            for batch_element in batch_state1:
                agent_input_list.append( batch_element["agents input state tensors"][agent.name] )
            agent_output = agent.step_tensor( torch.cat(agent_input_list) )
            agent_outputs.append(agent_output)
        agent_outputs_ten = torch.cat(agent_outputs, dim=1)
        critic_output_for_gradient = -critic.forward_tensor(critic_input_state1_ten.detach(), agent_outputs_ten)
        critic_loss_mean = critic_output_for_gradient.mean()
        critic_loss_mean.backward()
        for agent in agents:
            agent.optimizer_step()


        #
        # update target critic
        critic.update_target_network(TAU_TARGET_NETWORKS)

        #
        # update target network for actor
        for agent in agents:
            agent.update_target_network(TAU_TARGET_NETWORKS)

        #
        # store losses in the loss list
        output_lists["loss_list"].append({"Critic Loss": float(critic_loss_value.detach().numpy())})

        if timestep % 20 == 0:
            print(f"episode {episode_number:3}, timestep {timestep:5}: {state['time']}")


