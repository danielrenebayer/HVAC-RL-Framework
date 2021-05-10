
import os
import torch
import pickle
import timeit
import datetime
import numpy as np
from copy import deepcopy

from ReplayBuffer import ReplayBufferStd
import StateUtilities as SU

from Agents import agent_constructor
import RLCritics

def one_single_episode(algorithm,
        building, building_occ,
        agents = None, critics = None,
        hyper_params = None, episode_number = 0, sqloutput = None,
        extended_logging = False, evaluation_episode = False,
        add_random_process_in_eval_epoch = False,
        ts_diff_in_min = 5, rpb = None):
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
        TARGET_NETWORK_UPDATE_FREQ = 3
    else:
        LAMBDA_REWARD_ENERGY = hyper_params.lambda_rwd_energy
        LAMBDA_REWARD_MANU_STP_CHANGES = hyper_params.lambda_rwd_mstpc
        TAU_TARGET_NETWORKS  = hyper_params.tau
        DISCOUNT_FACTOR = hyper_params.discount_factor
        BATCH_SIZE      = hyper_params.batch_size
        RPB_BUFFER_SIZE = hyper_params.rpb_buffer_size
        LEARNING_RATE   = hyper_params.lr
        TARGET_NETWORK_UPDATE_FREQ = hyper_params.target_network_update_freq
    #
    # define the output dict containing status informations
    status_output_dict = {}
    if hyper_params.verbose_output_mode:
        status_output_dict["verbose_output"] = []
    #
    # Define the replay ReplayBuffer
    if rpb is None:
        rpb = ReplayBufferStd(size=RPB_BUFFER_SIZE, number_agents=len(agents))
    #
    # Define the loss for DDQN
    loss = torch.nn.MSELoss()
    #
    # Lists for command-line outputs
    reward_list = []
    output_loss_list = []
    output_q_st2_list= []
    output_J_mean_list=[]
    output_cr_frobnorm_mat_list = []
    output_cr_frobnorm_bia_list = []
    output_ag_frobnorm_mat_list = []
    output_ag_frobnorm_bia_list = []
    output_n_stp_ch  = []
    output_energy_Wh = []
    if not algorithm == "ddpg":
        output_q_st2_list= [0 for _ in agents]
        output_J_mean_list=[0 for _ in agents]
        output_cr_frobnorm_mat_list = [0 for _ in agents]
        output_cr_frobnorm_bia_list = [0 for _ in agents]
    if algorithm == "baseline_rule-based":
        output_loss_list = [0]
        output_ag_frobnorm_mat_list = [0]
        output_ag_frobnorm_bia_list = [0]
    # list for q values output, if selected
    if evaluation_episode and hyper_params.output_Q_vals_iep:
        q_values_list = [ [] for _ in agents ]
    #
    # prepare the simulation
    state = building.model_reset()
    SU.fix_year_confussion(state)
    norm_state_ten = SU.unnormalized_state_to_tensor(state, building)
    #
    current_occupancy = building_occ.draw_sample( state["time"] )
    timestep   = 0
    last_state = None
    # start the simulation loop
    while not building.model_is_terminate():
        actions = list()

        currdate = state['time']
        #
        # request occupancy for the next state
        nextdate = state['time'] + datetime.timedelta(minutes=ts_diff_in_min)
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
        add_random_process = True
        if evaluation_episode and not add_random_process_in_eval_epoch:
            add_random_process = False

        if algorithm == "ddqn":
            if agents[0].shared_network_per_agent_class:
                new_actions = agents[0].next_action(norm_state_ten, add_random_process)
                agent_actions_list = new_actions
                # decode the actions for every agent using the individual agent objects
                for idx, agent in enumerate(agents):
                    agent_actions_dict[agent.name] = agent.output_action_to_action_dict(new_actions[idx])
            else:
                for agent in agents:
                    new_action = agent.next_action(norm_state_ten, add_random_process)
                    agent_actions_list.append( new_action )
                    agent_actions_dict[agent.name] = agent.output_action_to_action_dict(new_action)
                    if hyper_params.verbose_output_mode:
                        _, vo_ipt = agent.step_tensor(norm_state_ten, True, True)
                        vodict = {"state": state, "norm_state_ten": norm_state_ten,
                                  "agent_action": new_action,
                                  "agent internal input tensor": vo_ipt.detach()}
                        status_output_dict["verbose_output"].append(vodict)
            # no backtransformation of variables needed, this is done in agents definition already
            #
            # output Q values in eval episode if selected
            if evaluation_episode and hyper_params.output_Q_vals_iep:
                for idx, agent in enumerate(agents):
                    q_values = agent.step_tensor(norm_state_ten, use_actor=True).detach().numpy()
                    q_values_list[idx].append(q_values)

        elif algorithm == "ddpg":
            for agent in agents:
                new_action = agent.step_tensor(norm_state_ten,
                                               use_actor = True,
                                               add_ou    = add_ou_process)
                agent_actions_list.append( new_action )
                new_action_dict = agent.output_tensor_to_action_dict(new_action)
                agent_actions_dict[agent.name] = SU.backtransform_variables_in_dict(new_action_dict, inplace=True)

        elif algorithm == "baseline_rule-based":
            for agent in agents:
                agent_actions_dict[agent.name] = agent.step(state)

        #
        # send agent actions to the building object and obtaion the actions for COBS/eplus
        actions.extend( building.obtain_cobs_actions( agent_actions_dict, state["timestep"]+1 ) )

        #
        # send actions to EnergyPlus and obtian the new state
        norm_state_ten_last = norm_state_ten
        last_state = state
        timestep  += 1
        state      = building.model_step(actions)
        current_occupancy = next_occupancy
        SU.fix_year_confussion(state)

        current_energy_Wh = state["energy"] / 360

        #
        # modify state
        norm_state_ten = SU.unnormalized_state_to_tensor(state, building)

        #
        # send current temp/humidity values for all rooms
        # obtain number of manual setpoint changes
        _, n_manual_stp_changes, target_temp_per_room = building_occ.manual_setpoint_changes(state['time'], state["temperature"], None, hyper_params.stp_reward_step_offset)

        #
        # reward computation
        if hyper_params is None or hyper_params.reward_function == "sum_energy_mstpc":
            n_manual_stp_changes_after_function = setpoint_activation_function(n_manual_stp_changes, hyper_params.stp_reward_function)
            reward = LAMBDA_REWARD_ENERGY * current_energy_Wh + LAMBDA_REWARD_MANU_STP_CHANGES * n_manual_stp_changes_after_function
        elif hyper_params.reward_function == "rulebased_roomtemp":
            reward, target_temp_per_room = reward_fn_rulebased_roomtemp(state, building, hyper_params.stp_reward_step_offset)
            reward = setpoint_activation_function(reward, hyper_params.stp_reward_function)
        #elif hyper_params.reward_function == "rulebased_agent_output":
        else:
            reward, target_temp_per_room = reward_fn_rulebased_agent_output(state, agent_actions_dict, building, hyper_params.stp_reward_step_offset)
            reward = setpoint_activation_function(reward, hyper_params.stp_reward_function)
        # invert and scale reward and (maybe) add offset
        reward = -hyper_params.reward_scale * reward + hyper_params.reward_offset
        if not hyper_params is None and hyper_params.log_reward:
            reward = - np.log(-reward + 1)
        # add reward to output list for command-line outputs
        reward_list.append(reward)
        output_n_stp_ch.append(n_manual_stp_changes)
        output_energy_Wh.append(current_energy_Wh)

        #
        # save (last_state, actions, reward, state) to replay buffer
        rpb.add_transition(norm_state_ten_last, agent_actions_list, reward, norm_state_ten)

        #
        if algorithm == "ddqn":
            # sample minibatch
            b_state1, b_action, b_reward, b_state2 = rpb.sample_minibatch(BATCH_SIZE, False)
            b_action = torch.tensor(b_action)
            #
            # loop over all [agent, critic]-pairs
            if agents[0].shared_network_per_agent_class:
                #
                # compute y (i.e. the TD-target)
                #  Hint: s_{i+1} <- state2; s_i <- state1
                agents[0].model_actor.zero_grad()
                b_reward = b_reward.detach().expand(-1, len(agents) ).flatten()[:, np.newaxis]
                # wrong: b_reward = b_reward.detach().repeat(len(agents), 1)
                y = b_reward + DISCOUNT_FACTOR * agents[0].step_tensor(b_state2, use_actor = False).detach().max(dim=1).values[:, np.newaxis]
                # compute Q for state1
                q = agents[0].step_tensor(b_state1, use_actor = True).gather(1, b_action.flatten()[:, np.newaxis])
                # update agent by minimizing the loss L
                L = loss(q, y)
                L.backward()
                agents[0].optimizer_step()
                #
                # save outputs
                output_loss_list.append(float(L.detach().numpy()))
                # compute and store frobenius norms for the weights
                ag_fnorm1, ag_fnorm2 = 0, 0
                for p in agents[0].model_actor.parameters():
                    if len(p.shape) == 1: ag_fnorm2 += float(p.norm().detach().cpu().numpy())
                    else: ag_fnorm1 += float(p.norm().detach().cpu().numpy())
                output_ag_frobnorm_mat_list.append( ag_fnorm1 )
                output_ag_frobnorm_bia_list.append( ag_fnorm2 )
            else:
                for agent_id, agent in enumerate(agents):
                    #
                    # compute y (i.e. the TD-target)
                    #  Hint: s_{i+1} <- state2; s_i <- state1
                    agent.model_actor.zero_grad()
                    y = b_reward.detach() + DISCOUNT_FACTOR * agent.step_tensor(b_state2, use_actor = False).detach().max(dim=1).values[:, np.newaxis]
                    # compute Q for state1
                    q = agent.step_tensor(b_state1, use_actor = True).gather(1, b_action[:, agent_id][:, np.newaxis])
                    # update agent by minimizing the loss L
                    L = loss(q, y)
                    L.backward()
                    agent.optimizer_step()
                    #
                    # save outputs
                    output_loss_list.append(float(L.detach().numpy()))
                    # compute and store frobenius norms for the weights
                    ag_fnorm1, ag_fnorm2 = 0, 0
                    for p in agent.model_actor.parameters():
                        if len(p.shape) == 1: ag_fnorm2 += float(p.norm().detach().cpu().numpy())
                        else: ag_fnorm1 += float(p.norm().detach().cpu().numpy())
                    output_ag_frobnorm_mat_list.append( ag_fnorm1 )
                    output_ag_frobnorm_bia_list.append( ag_fnorm2 )


        elif algorithm == "ddpg":
            # sample minibatch
            b_state1, b_action, b_action_cat, b_reward, b_state2 = rpb.sample_minibatch(BATCH_SIZE)
            #
            # loop over all [agent, critic]-pairs
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
                L = critic.compute_loss_and_optimize(q, y, no_backprop = evaluation_episode)
                #
                # update actor policies
                # policy loss = J
                mu_list = [ aInnerLoop.step_tensor(b_state1, add_ou = False) for aInnerLoop in agents ]
                agent.model_actor.zero_grad()
                policy_J = -critic.forward_tensor(b_state1, mu_list)
                policy_J_mean = policy_J.mean()
                if not evaluation_episode:
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

        #
        # store detailed output, if extended logging is selected
        if extended_logging and not sqloutput is None:
            sqloutput.add_every_step_of_some_episodes( locals() )

        if timestep % 200 == 0:
            eval_ep_str     = "  " if evaluation_episode else "no"
            rand_pr_add_str = "  " if add_random_process else "no"
            print(f"ep. {episode_number:3}, ts. {timestep:5}: {state['time']}, {eval_ep_str} eval ep., {rand_pr_add_str} rand. p. add.")

    #
    # update target networks
    status_output_dict["target_network_update"] = False
    if episode_number % TARGET_NETWORK_UPDATE_FREQ == 0:
        if algorithm == "ddqn":
            for agent in agents:
                agent.copy_weights_to_target()
            status_output_dict["target_network_update"] = True
        elif algorithm == "ddpg":
            # update target critic
            for critic in critics:
                critic.update_target_network(TAU_TARGET_NETWORKS)
            # update target network for actor
            for agent in agents:
                agent.update_target_network(TAU_TARGET_NETWORKS)
            status_output_dict["target_network_update"] = True

    #
    # status output dict postprocessing
    status_output_dict["episode"]     = episode_number
    status_output_dict["lr"]          = LEARNING_RATE
    status_output_dict["tau"]         = TAU_TARGET_NETWORKS
    status_output_dict["lambda_energy"] = LAMBDA_REWARD_ENERGY
    status_output_dict["lambda_manu_stp"] = LAMBDA_REWARD_MANU_STP_CHANGES
    status_output_dict["reward_mean"] = np.mean(reward_list)
    status_output_dict["reward_sum"]  = np.sum(reward_list)
    status_output_dict["sum_manual_stp_ch_n"]    = np.sum(output_n_stp_ch)
    status_output_dict["mean_manual_stp_ch_n"]   = np.mean(output_n_stp_ch)
    status_output_dict["current_energy_Wh_mean"] = np.mean(output_energy_Wh)
    status_output_dict["current_energy_Wh_sum"]  = np.sum(output_energy_Wh)
    status_output_dict["evaluation_epoch"] = evaluation_episode
    status_output_dict["random_process_addition"] = add_random_process
    status_output_dict["loss_mean"] = np.mean(output_loss_list)
    status_output_dict["frobnorm_agent_matr_mean"] = np.mean(output_ag_frobnorm_mat_list)
    status_output_dict["frobnorm_agent_bias_mean"] = np.mean(output_ag_frobnorm_bia_list)
    if algorithm == "ddpg":
        status_output_dict["frobnorm_critic_matr_mean"] = np.mean(output_cr_frobnorm_mat_list)
        status_output_dict["frobnorm_critic_bias_mean"] = np.mean(output_cr_frobnorm_bia_list)
        status_output_dict["q_st2_mean"] = np.mean(output_q_st2_list)
        status_output_dict["J_mean"] = np.mean(output_J_mean_list)
    # output q values list if selected
    if evaluation_episode and hyper_params.output_Q_vals_iep:
        f = open(os.path.join(hyper_params.checkpoint_dir, f"q_values.pickle"), "wb")
        pickle_dict = {
            "Q value list": q_values_list,
            "Actions":      [ [agent.output_action_to_action_dict(i) for i in range(len(agent.output_to_action_mapping))] for agent in agents ]
        }
        pickle.dump(pickle_dict, f)
        f.close()
        #f = os.path.join(hyper_params.checkpoint_dir, f"q_values")
        #np.save(f, np.array(q_values_list))
    return status_output_dict





def run_for_n_episodes(n_episodes, building, building_occ, args, sqloutput = None, episode_offset = 0):
    """
    Runs the ddpg algorithm (i.e. the above defined ddpg_episode_mc function)
    for n_episodes runs.
    The agents and critics will be initialized according to the building object.
    """

    #
    # prepair the load of existing models if selected
    load_path    = ""
    load_episode = 0
    if args.load_models_from_path != "":
        load_episode = args.load_models_episode
        load_path    = os.path.abspath(args.load_models_from_path)

    #
    # Define the agents
    agents = []
    idx    = 0
    # HINT: a device can be a zone, too
    for agent_name, (controlled_device, controlled_device_type) in building.agent_device_pairing.items():
        new_agent = agent_constructor( controlled_device_type )
        new_agent.initialize(
                         name = agent_name,
                         args = args,
                         controlled_element = controlled_device,
                         global_state_keys  = building.global_state_variables,
                         load_path          = load_path,  # loads the model from this path, if it is not empty
                         load_prefix        = f"episode_{load_episode}_agent_{idx}")
        agents.append(new_agent)
        idx += 1

    #
    # Define the critics
    critics = []
    if args.algorithm == "ddpg":
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
        # Load existing models if selected
        if args.load_models_from_path != "":
            for idx, critic in enumerate(critics):
                critic.load_models_from_disk(load_path, prefix=f"episode_{load_episode}_critic_{idx}")
                print(f"Critic {idx} loaded from {load_path}")

    #
    # initialize the replay buffer here, to get it over the episodes
    rpb = ReplayBufferStd(size=args.rpb_buffer_size, number_agents=len(agents))

    #
    # Set model parameters
    episode_len        = args.episode_length
    episode_start_day  = args.episode_start_day
    episode_start_month= args.episode_start_month
    ts_diff_in_min     = 60 // args.ts_per_hour
    building.model.set_runperiod(episode_len, 2017, episode_start_month, episode_start_day)
    building.model.set_timestep( args.ts_per_hour )

    if args.algorithm == "baseline_rule-based":
        status_output_dict = \
            one_single_episode(
                        algorithm      = "baseline_rule-based",
                        building       = building,
                        building_occ   = building_occ,
                        agents         = agents,
                        critics        = critics,
                        hyper_params   = args,
                        episode_number = 0,
                        sqloutput      = sqloutput,
                        extended_logging   = True,
                        evaluation_episode = True,
                        add_random_process_in_eval_epoch = False,
                        ts_diff_in_min = ts_diff_in_min,
                        rpb            = rpb)
        if not sqloutput is None:
            sqloutput.add_last_step_of_episode( status_output_dict )
        return

    for n_episode in range(episode_offset, n_episodes + episode_offset):

        t_start = timeit.default_timer()

        if args.algorithm == "ddqn":
            # set epsilon for all agents
            epsilon = max(args.epsilon, np.exp(n_episode * np.log(args.epsilon)/args.epsilon_final_step))
            for agent in agents:
                agent.epsilon = epsilon
        elif args.algorithm == "ddpg":
            # set ou-parameters for all agents
            ou_mu    = 0.0
            ou_theta = max(args.ou_theta, np.exp(n_episode * np.log(args.ou_theta / 3) / args.epsilon_final_step))
            ou_sigma = max(args.ou_sigma, np.exp(n_episode * np.log(args.ou_sigma / 3) / args.epsilon_final_step))
            for agent in agents:
                agent.ou_theta = ou_theta
                agent.ou_sigma = ou_sigma
                agent.ou_mu    = ou_mu

        # run one episode
        status_output_dict = \
            one_single_episode(
                        algorithm      = args.algorithm,
                        building       = building,
                        building_occ   = building_occ,
                        agents         = agents,
                        critics        = critics,
                        hyper_params   = args,
                        episode_number = n_episode,
                        sqloutput      = sqloutput,
                        extended_logging   = (n_episode+1) % args.network_storage_frequency == 0,
                        evaluation_episode = (n_episode+1) % args.network_storage_frequency == 0,
                        add_random_process_in_eval_epoch = args.add_ou_in_eval_epoch,
                        ts_diff_in_min = ts_diff_in_min,
                        rpb = rpb)

        if args.algorithm == "ddqn":
            status_output_dict["epsilon"] = epsilon
        elif args.algorithm == "ddpg":
            status_output_dict["ou_theta"] = ou_theta
            status_output_dict["ou_sigma"] = ou_sigma
            status_output_dict["ou_mu"]    = ou_mu

        if args.verbose_output_mode:
            f = open(f"{args.checkpoint_dir}/vbo-status_output_dict-{n_episode}.pickle", "wb")
            pickle.dump(status_output_dict["verbose_output"], f)
            f.close()
            f = open(f"{args.checkpoint_dir}/vbo-agents-{n_episode}.pickle", "wb")
            pickle.dump(agents, f)
            f.close()
            f = open(f"{args.checkpoint_dir}/vbo-building-{n_episode}.pickle", "wb")
            pickle.dump(building.global_state_variables, f)
            f.close()

        t_end  = timeit.default_timer()
        t_diff = t_end - t_start
        status_output_dict["t_diff"]  = t_diff
        print(f"Episode {n_episode:5} finished: mean reward = {status_output_dict['reward_mean']:9.4f}, ", end="")
        if args.algorithm == "ddqn": print(f"epsilon = {epsilon:.4f}, ", end="")
        print(f"time = {t_diff:6.1f}s, random process = {status_output_dict['random_process_addition']}, eval. epoch = {status_output_dict['evaluation_epoch']}, target w. upd. = {status_output_dict['target_network_update']}")
        # store detailed output
        if not sqloutput is None:
            sqloutput.add_last_step_of_episode( status_output_dict )

        # save agent/critic networks every selected run
        if (n_episode+1) % args.network_storage_frequency == 0:
            for idx, agent in enumerate(agents):
                agent.save_models_to_disk(args.checkpoint_dir, prefix=f"episode_{n_episode}_agent_{idx}")
            if args.algorithm == "ddpg":
                for idx, critic in enumerate(critics):
                    critic.save_models_to_disk(args.checkpoint_dir, prefix=f"episode_{n_episode}_critic_{idx}")

        # commit sql output if available
        if not sqloutput is None: sqloutput.db.commit()




def setpoint_activation_function(n_stp_changes, function_name):
    if function_name == "quadratic":
        return np.power(n_stp_changes, 2)
    elif function_name == "cubic":
        return np.power(n_stp_changes, 3)
    elif function_name == "exponential":
        return np.exp(n_stp_changes)
    elif function_name == "linear":
        return n_stp_changes
    else:
        raise AttributeError(f"Unknown function name {function_name}.")





def reward_fn_rulebased_roomtemp(state, building, discomfort_step_offset = 0.0):
    """
    This is an alternative reward function.
    It loops over all rooms, returning a (positiv) reward, iff the temperature is out of a given band.
    The band changes between office hours (Mo - Fr, 7.00 - 18.00) and the rest of the time.
    """
    changed_magnitude = 0
    dto = state['time']
    temp_values = state['temperature']
    target_temp_per_room = {}
    for room in building.room_names:
        if dto.weekday() < 5 and dto.hour >= 7 and dto.hour < 18:
            # if the temperature is not in the range [21.0,23.0], change the setpoint
            target_temp_per_room[room] = 22.0
            if temp_values[room] < 21.0:
                changed_magnitude += 21.0 - temp_values[room]
                changed_magnitude += discomfort_step_offset
            elif temp_values[room] > 23.5:
                changed_magnitude += temp_values[room] - 23.5
                changed_magnitude += discomfort_step_offset
        else:
            # if the temperature is not in the range [15.0,17.0], change the setpoint
            target_temp_per_room[room] = 16.0
            if temp_values[room] < 15:
                changed_magnitude += 15.0 - temp_values[room]
                changed_magnitude += discomfort_step_offset
            elif temp_values[room] > 17:
                changed_magnitude += temp_values[room] - 17.0
                changed_magnitude += discomfort_step_offset
    return changed_magnitude, target_temp_per_room




def reward_fn_rulebased_agent_output(state, agent_actions_dict, building, discomfort_step_offset = 0.0):
    """
    This is another alternative reward function.
    It loops over all agents, returning a (positiv) reward, iff the agent-computed heating setpoint is out of a given band.
    The band changes between office hours (Mo - Fr, 7.00 - 18.00) and the rest of the time.
    """
    changed_magnitude = 0
    dto = state['time']
    target_temp_per_room = {}
    for agent, agent_actions in agent_actions_dict.items():
        if "Zone Heating/Cooling-Mean Setpoint" in agent_actions.keys():
            agent_heating_setpoint = agent_actions["Zone Heating/Cooling-Mean Setpoint"] - agent_actions["Zone Heating/Cooling-Delta Setpoint"]
        else:
            agent_heating_setpoint = agent_actions["Zone Heating Setpoint"]
        if dto.weekday() < 5 and dto.hour >= 7 and dto.hour < 18:
            # if the temperature is not in the range [21.0,23.0], change the setpoint
            for room in building.room_names:
                target_temp_per_room[room] = 22.0
            if agent_heating_setpoint < 21.0:
                changed_magnitude += 21.0 - agent_heating_setpoint
                changed_magnitude += discomfort_step_offset
            elif agent_heating_setpoint > 23.5:
                changed_magnitude += agent_heating_setpoint - 23.5
                changed_magnitude += discomfort_step_offset
        else:
            # if the temperature is not in the range [15,17], change the setpoint
            for room in building.room_names:
                target_temp_per_room[room] = 16.0
            if agent_heating_setpoint < 15:
                changed_magnitude += 15.0 - agent_heating_setpoint
                changed_magnitude += discomfort_step_offset
            elif agent_heating_setpoint > 17:
                changed_magnitude += agent_heating_setpoint - 17.0
                changed_magnitude += discomfort_step_offset
    return changed_magnitude, target_temp_per_room



