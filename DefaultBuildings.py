"""
This file contains the default buildings, that can be used in EnergyPlus using the COBS API
"""

import os
import cobs
from copy import deepcopy
from global_paths import global_paths

class Building_5ZoneAirCooled:
    def __init__(self, args = None):
        #
        #
        self.room_names = [f'SPACE{i}-1' for i in range(1,6)]
        #
        # the pairing which gives, which agent (identified by name) controlles which device (or zone) and what kind of device it is
        if args.algorithm == "ddpg":
            self.agent_device_pairing = {f"Agent SPACE{i}-1":
                                         (f"SPACE{i}-1", "VAV with Reheat,Heating,Cooling,RL")
                                     for i in range(1,6)}
        elif args.algorithm == "ddqn":
            self.agent_device_pairing = {f"Agent SPACE{i}-1":
                                         (f"SPACE{i}-1", "VAV with Reheat,Heating,Cooling,Q,RL")
                                     for i in range(1,6)}
        elif args.algorithm == "rule-based":
            self.agent_device_pairing = {f"Agent SPACE{i}-1":
                                         (f"SPACE{i}-1", "VAV with Reheat,Heating,Cooling,NoRL")
                                     for i in range(1,6)}
        #
        # A dictionary formatted as {target sensor: state name (as named in the output state dict)}
        # Target sensor is a tuple: (variable name, key value)
        self.eplus_extra_states = {}
        # dict fro the variables, that are included as extra states, but do not have a specific key (i.e. "*" as key value)
        self.eplus_var_types    = {}
        #
        # add the Air Terminal VAV Damper Positions
        for r in self.room_names:
            self.eplus_extra_states[("Zone Air Terminal VAV Damper Position", f"{r} VAV Reheat")] = f"{r} Zone VAV Reheat Damper Position"
            #self.eplus_extra_states[("Schedule Value", "SPACE1-1 VAV Customized Schedule")] = "VAV Setpoint"
        #
        # add CO2-Concentration Simulation
        for r in self.room_names:
            self.eplus_extra_states[("Zone Air CO2 Concentration", r)] = f"{r} Zone CO2"
        #
        # add state output values for zone people occupant count
        for r in self.room_names:
            self.eplus_extra_states[("Zone People Occupant Count", r)] = f"{r} Zone People Count"
        #
        # add state output values for relative air humidity
        for r in self.room_names:
            self.eplus_extra_states[("Zone Air Relative Humidity", r)] = f"{r} Zone Relative Humidity"
        #
        # add outdoor air temperature, humidity, wind and solar radiation settings (from weather file)
        self.eplus_extra_states[("Site Outdoor Air Drybulb Temperature", "*")] = "Outdoor Air Temperature"
        self.eplus_extra_states[("Site Outdoor Air Relative Humidity",   "*")] = "Outdoor Air Humidity"
        self.eplus_extra_states[('Site Wind Speed',     '*')] = "Outdoor Wind Speed"
        self.eplus_extra_states[('Site Wind Direction', '*')] = "Outdoor Wind Direction"
        self.eplus_extra_states[('Site Diffuse Solar Radiation Rate per Area', '*')] = "Outdoor Solar Radi Diffuse"
        self.eplus_extra_states[('Site Direct Solar Radiation Rate per Area',  '*')] = "Outdoor Solar Radi Direct"
        self.eplus_var_types['Site Outdoor Air Drybulb Temperature'] = "Environment"
        self.eplus_var_types['Site Outdoor Air Relative Humidity']   = "Environment"
        self.eplus_var_types['Site Wind Speed']     = "Environment"
        self.eplus_var_types['Site Wind Direction'] = "Environment"
        self.eplus_var_types['Site Diffuse Solar Radiation Rate per Area'] = "Environment"
        self.eplus_var_types['Site Direct Solar Radiation Rate per Area']  = "Environment"
        #
        # parse the file paths for IDF and EPW file
        idf_file_path = os.path.join(global_paths["COBS"], "cobs/data/buildings/5ZoneAirCooled.idf")
        epw_file_path = os.path.join(global_paths["COBS"], "cobs/data/weathers/USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw")
        if not args is None:
            if args.idf_file != "": idf_file_path = os.path.abspath(args.idf_file)
            if args.epw_file != "": epw_file_path = os.path.abspath(args.epw_file)
        #
        # define the model
        model = cobs.Model(
            idf_file_name = idf_file_path,
            weather_file  = epw_file_path,
            #weather_file  = os.path.join(global_paths["COBS"], "cobs/data/weathers/1.epw"),
            eplus_naming_dict = self.eplus_extra_states,
            eplus_var_types   = self.eplus_var_types,
            tmp_idf_path      = None if args is None else args.checkpoint_dir
        )
        #
        # Modify the model, if arguments are given
        if not args is None:
            idx = model.run_parameters.index("-d")
            model.run_parameters[idx + 1] = os.path.join(args.checkpoint_dir, "result")
        #
        # Save all model outputs, if args.eplus_storage_mode is given
        self.storage_mode = False
        if not args is None and args.eplus_storage_mode:
            self.storage_mode = True
            self._storage_list_full = False
            self._storage_next_idx  = 0
            self.storage_list = []
        #
        #
        # generate edd file (which lists the possible actions):
        #if not os.path.isfile("./result/eplusout.edd"):
        if not model.get_configuration("Output:EnergyManagementSystem"):
            model.add_configuration("Output:EnergyManagementSystem",
                            values={"Actuator Availability Dictionary Reporting": "Verbose",
                                    "Internal Variable Availability Dictionary Reporting": "Verbose",
                                    "EMS Runtime Language Debug Output Level": "ErrorsOnly"})
        #
        #
        # Add these sensing option into the IDF file (optional, if IDF already contains them, you can ignore it)
        for key, _ in self.eplus_extra_states.items():
            model.add_configuration("Output:Variable",
                                    {"Key Value": key[1], "Variable Name": key[0], "Reporting Frequency": "Timestep"})
        #
        # Because the SPACE_-1 VAV Customized Schedule does not exist, add it to the model
        for r in self.room_names:
            model.add_configuration("Schedule:Constant",
                                   {"Name": f"{r} VAV Customized Schedule",
                                    "Schedule Type Limits Name": "Fraction",
                                    "Hourly Value": 0})
            # Overwrite existing VAV control policy to gain customized control
            model.edit_configuration(
                idf_header_name="AirTerminal:SingleDuct:VAV:Reheat",
                identifier={"Name": f"{r} VAV Reheat"},
                update_values={"Zone Minimum Air Flow Input Method": "Scheduled",
                               "Minimum Air Flow Fraction Schedule Name": f"{r} VAV Customized Schedule"})
        #
        # Add simulation of CO2 concentration. It is not included in 5ZoneAirCooled.idf, so I add it here
        # You have to provide the outdoor CO2 concentration in order to let Energyplus calculate the indoor CO2
        # Here I use a fixed 410.25 ppm as the outdoor CO2 concentration
        model.add_configuration("Schedule:Constant",
                                {"Name": "Outdoor CO2 Schedule",
                                 "Schedule Type Limits Name": "Any Number",
                                 "Hourly Value": 410.25})
        model.add_configuration("ZoneAirContaminantBalance",
                               {"Carbon Dioxide Concentration": "Yes",
                                "Outdoor Carbon Dioxide Schedule Name": "Outdoor CO2 Schedule"})
        #
        # get the full control over the occupancy at each time step
        for r in self.room_names:
            model.add_configuration("Schedule:Constant",
                                    {"Name": f"OCC-SCHEDULE-{r}",
                                     "Schedule Type Limits Name": "Fraction",
                                     "Hourly Value": 0.0})
            model.edit_configuration("People",
                                    {"Name": f"{r} People 1"},
                                    {"Number of People Schedule Name": f"OCC-SCHEDULE-{r}",
                                     "Number of People": 50})
        #
        # list all state variables
        self.global_state_variables = ["Minutes of Day", "Day of Week", "Calendar Week"]
        for r in self.room_names:
            self.global_state_variables.append(f"{r} Zone Temperature")
        for _, varname in self.eplus_extra_states.items():
            self.global_state_variables.append(varname)
        #
        # make model a instance variable
        self.model = model
        self.cobs_model = model

    def obtain_cobs_actions(self, agent_actions, next_timestep):
        """
        Returns a list of actions that can be passed to cobs.

        The agent_actions dict is expected to have the format:
        {agent_name: {controlled_parameter: new_value}}
        """
        actions = []
        for agent_name, ag_actions in agent_actions.items():
            controlled_group, _ = self.agent_device_pairing[agent_name]
            if "Zone VAV Reheat Damper Position" in ag_actions.keys():
                action_val = ag_actions["Zone VAV Reheat Damper Position"]
                actions.append({"priority": 0,
                                "component_type": "Schedule:Constant",
                                "control_type": "Schedule Value",
                                "actuator_key": f"{controlled_group} VAV Customized Schedule",
                                "value": action_val,
                                "start_time": next_timestep})
            if "Zone Heating/Cooling-Mean Setpoint"  in ag_actions.keys() and \
            "Zone Heating/Cooling-Delta Setpoint" in ag_actions.keys():
                mean_temp_sp = ag_actions["Zone Heating/Cooling-Mean Setpoint"]
                delta = ag_actions["Zone Heating/Cooling-Delta Setpoint"]
                if delta < 0.1: delta = 0.1
                actions.append({"value":      mean_temp_sp - delta,
                                "start_time": next_timestep,
                                "priority":   0,
                                "component_type": "Zone Temperature Control",
                                "control_type":   "Heating Setpoint",
                                "actuator_key":   controlled_group})
                actions.append({"value":      mean_temp_sp + delta,
                                "start_time": next_timestep,
                                "priority":   0,
                                "component_type": "Zone Temperature Control",
                                "control_type":   "Cooling Setpoint",
                                "actuator_key":   controlled_group})
            # ... das macht jetzt keinen Sinn mehr, eine Prüfung wäre aber vielleicht nicht schlecht
            #else:
            #    print(f"Action {action_name} from agent in zone {zone} unknown.")
        return actions

    def model_step(self, actions):
        """
        Sends the current actions to the model and obtains the new state.
        If storage_mode is active (by args.eplus_storage_mode), it
        use the informations from the storage, if model_reset() has be callen
        after there was content in the storage buffer.
        """
        if self.storage_mode:
            if self._storage_list_full:
                newstate = self.storage_list[ self._storage_next_idx ]
                self._storage_next_idx += 1
                return newstate
            else:
                newstate = self.model.step(actions)
                self.storage_list.append(deepcopy(newstate))
                return newstate
        else:
            return self.model.step(actions)

    def model_is_terminate(self):
        if self.storage_mode and self._storage_list_full:
            return self._storage_next_idx >= len(self.storage_list)
        else:
            return self.model.is_terminate()

    def model_reset(self):
        if self.storage_mode:
            if self._storage_list_full:
                self._storage_next_idx  = 0
                return self.model_step( [] )
            elif len(self.storage_list) > 0:
                self._storage_list_full = True
                self._storage_next_idx  = 0
                return self.model_step( [] )
            else:
                first_state = self.model.reset()
                self.storage_list.append(deepcopy(first_state))
                return first_state
        else:
            return self.model.reset()

class Building_5ZoneAirCooled_SmallAgents(Building_5ZoneAirCooled):
    def __init__(self, args = None):
        #
        super().__init__(args)
        #
        if args.algorithm == "ddpg":
            self.agent_device_pairing = {f"Agent SPACE{i}-1":
                                         (f"SPACE{i}-1", "VAV with Reheat,Heating,Cooling,RL,VerySmall")
                                     for i in range(1,6)}
        elif args.algorithm == "ddqn":
            self.agent_device_pairing = {f"Agent SPACE{i}-1":
                                         (f"SPACE{i}-1", "VAV with Reheat,Heating,Cooling,Q,RL,VerySmall")
                                     for i in range(1,6)}

    def obtain_cobs_actions(self, agent_actions, next_timestep):
        """
        Returns a list of actions that can be passed to cobs.

        The agent_actions dict is expected to have the format:
        {agent_name: {controlled_parameter: new_value}}
        """
        actions = []
        for agent_name, ag_actions in agent_actions.items():
            controlled_group, _ = self.agent_device_pairing[agent_name]
            #
            # set VAV constantly to 1
            actions.append({"priority": 0,
                            "component_type": "Schedule:Constant",
                            "control_type": "Schedule Value",
                            "actuator_key": f"{controlled_group} VAV Customized Schedule",
                            "value": 1,
                            "start_time": next_timestep})
            #
            mean_temp_sp = ag_actions["Zone Heating/Cooling-Mean Setpoint"]
            delta = ag_actions["Zone Heating/Cooling-Delta Setpoint"]
            if delta < 0.1: delta = 0.1
            actions.append({"value":      mean_temp_sp - delta,
                            "start_time": next_timestep,
                            "priority":   0,
                            "component_type": "Zone Temperature Control",
                            "control_type":   "Heating Setpoint",
                            "actuator_key":   controlled_group})
            actions.append({"value":      mean_temp_sp + delta,
                            "start_time": next_timestep,
                            "priority":   0,
                            "component_type": "Zone Temperature Control",
                            "control_type":   "Cooling Setpoint",
                            "actuator_key":   controlled_group})
            # ... das macht jetzt keinen Sinn mehr, eine Prüfung wäre aber vielleicht nicht schlecht
            #else:
            #    print(f"Action {action_name} from agent in zone {zone} unknown.")
        return actions

class Building_5ZoneAirCooled_SingleSetpoint_SmallAgents(Building_5ZoneAirCooled):
    def __init__(self, args = None):
        #
        super().__init__(args)
        #
        if args.algorithm == "ddqn":
            self.agent_device_pairing = {f"Agent SPACE{i}-1":
                                         (f"SPACE{i}-1", "SingleSetpoint,Q,RL,VerySmall")
                                     for i in range(1,6)}
        else:
            raise AttributeError(f"{args.algorithm} is not available for the class Building_5ZoneAirCooled_SingleSetpoint_SmallAgents")

    def obtain_cobs_actions(self, agent_actions, next_timestep):
        """
        Returns a list of actions that can be passed to cobs.

        The agent_actions dict is expected to have the format:
        {agent_name: {controlled_parameter: new_value}}
        """
        actions = []
        for agent_name, ag_actions in agent_actions.items():
            controlled_group, _ = self.agent_device_pairing[agent_name]
            #
            # set VAV constantly to 1
            actions.append({"priority": 0,
                            "component_type": "Schedule:Constant",
                            "control_type": "Schedule Value",
                            "actuator_key": f"{controlled_group} VAV Customized Schedule",
                            "value": 1,
                            "start_time": next_timestep})
            #
            heating_temp   = ag_actions["Zone Heating Setpoint"]
            cooling_offset = 12
            actions.append({"value":      heating_temp,
                            "start_time": next_timestep,
                            "priority":   0,
                            "component_type": "Zone Temperature Control",
                            "control_type":   "Heating Setpoint",
                            "actuator_key":   controlled_group})
            actions.append({"value":      heating_temp + cooling_offset,
                            "start_time": next_timestep,
                            "priority":   0,
                            "component_type": "Zone Temperature Control",
                            "control_type":   "Cooling Setpoint",
                            "actuator_key":   controlled_group})
        return actions

class Building_5ZoneAirCooled_SingleSetpoint_SingleAgent(Building_5ZoneAirCooled):
    def __init__(self, args = None):
        #
        super().__init__(args)
        #
        if args.algorithm == "ddqn":
            self.agent_device_pairing = {"MainAgent":
                                             ("all", "SingleSetpoint,SingleAgent,Q,RL")
                                        }
        else:
            raise AttributeError(f"{args.algorithm} is not available for the class Building_5ZoneAirCooled_SingleSetpoint_SingleSmallAgent")

    def obtain_cobs_actions(self, agent_actions, next_timestep):
        """
        Returns a list of actions that can be passed to cobs.

        The agent_actions dict is expected to have the format:
        {agent_name: {controlled_parameter: new_value}}
        """
        actions = []
        for agent_name, ag_actions in agent_actions.items():
            # there is just one agent, so this loop is executed once only
            for zone in self.room_names:
                #
                # set VAV constantly to 1
                actions.append({"priority": 0,
                                "component_type": "Schedule:Constant",
                                "control_type": "Schedule Value",
                                "actuator_key": f"{zone} VAV Customized Schedule",
                                "value": 1,
                                "start_time": next_timestep})
                #
                heating_temp   = ag_actions["Zone Heating Setpoint"]
                cooling_offset = 12
                actions.append({"value":      heating_temp,
                                "start_time": next_timestep,
                                "priority":   0,
                                "component_type": "Zone Temperature Control",
                                "control_type":   "Heating Setpoint",
                                "actuator_key":   zone})
                actions.append({"value":      heating_temp + cooling_offset,
                                "start_time": next_timestep,
                                "priority":   0,
                                "component_type": "Zone Temperature Control",
                                "control_type":   "Cooling Setpoint",
                                "actuator_key":   zone})
        return actions


class Building_5ZoneAirCooled_SingleSetpoint_SingleSmallAgent(Building_5ZoneAirCooled):
    def __init__(self, args = None):
        #
        super().__init__(args)
        #
        if args.algorithm == "ddqn":
            self.agent_device_pairing = {"MainAgent":
                                             ("all", "SingleSetpoint,SingleAgent,Q,RL,VerySmall2")
                                        }
        elif args.algorithm == "ddpg":
            self.agent_device_pairing = {"MainAgent":
                                            ("all", "5ZoneAirCooled,SingleAgent,SingleSetpoint,RL,VerySmall")
                                        }
        else:
            raise AttributeError(f"{args.algorithm} is not available for the class Building_5ZoneAirCooled_SingleSetpoint_SingleSmallAgent")

    def obtain_cobs_actions(self, agent_actions, next_timestep):
        """
        Returns a list of actions that can be passed to cobs.

        The agent_actions dict is expected to have the format:
        {agent_name: {controlled_parameter: new_value}}
        """
        actions = []
        for agent_name, ag_actions in agent_actions.items():
            # there is just one agent, so this loop is executed once only
            for zone in self.room_names:
                #
                # set VAV constantly to 1
                actions.append({"priority": 0,
                                "component_type": "Schedule:Constant",
                                "control_type": "Schedule Value",
                                "actuator_key": f"{zone} VAV Customized Schedule",
                                "value": 1,
                                "start_time": next_timestep})
                #
                heating_temp   = ag_actions["Zone Heating Setpoint"]
                cooling_offset = 12
                actions.append({"value":      heating_temp,
                                "start_time": next_timestep,
                                "priority":   0,
                                "component_type": "Zone Temperature Control",
                                "control_type":   "Heating Setpoint",
                                "actuator_key":   zone})
                actions.append({"value":      heating_temp + cooling_offset,
                                "start_time": next_timestep,
                                "priority":   0,
                                "component_type": "Zone Temperature Control",
                                "control_type":   "Cooling Setpoint",
                                "actuator_key":   zone})
        return actions

class Building_5ZoneAirCooled_SingleSetpoint(Building_5ZoneAirCooled):
    def __init__(self, args = None):
        #
        super().__init__(args)
        #
        if args.algorithm == "ddqn":
            self.agent_device_pairing = {f"Agent SPACE{i}-1":
                                         (f"SPACE{i}-1", "SingleSetpoint,Q,RL")
                                     for i in range(1,6)}
        else:
            raise AttributeError(f"{args.algorithm} is not available for the class Building_5ZoneAirCooled_SingleSetpoint")

    def obtain_cobs_actions(self, agent_actions, next_timestep):
        """
        Returns a list of actions that can be passed to cobs.

        The agent_actions dict is expected to have the format:
        {agent_name: {controlled_parameter: new_value}}
        """
        actions = []
        for agent_name, ag_actions in agent_actions.items():
            controlled_group, _ = self.agent_device_pairing[agent_name]
            #
            # set VAV constantly to 1
            actions.append({"priority": 0,
                            "component_type": "Schedule:Constant",
                            "control_type": "Schedule Value",
                            "actuator_key": f"{controlled_group} VAV Customized Schedule",
                            "value": 1,
                            "start_time": next_timestep})
            #
            heating_temp   = ag_actions["Zone Heating Setpoint"]
            cooling_offset = 12
            actions.append({"value":      heating_temp,
                            "start_time": next_timestep,
                            "priority":   0,
                            "component_type": "Zone Temperature Control",
                            "control_type":   "Heating Setpoint",
                            "actuator_key":   controlled_group})
            actions.append({"value":      heating_temp + cooling_offset,
                            "start_time": next_timestep,
                            "priority":   0,
                            "component_type": "Zone Temperature Control",
                            "control_type":   "Cooling Setpoint",
                            "actuator_key":   controlled_group})
        return actions

class Building_5ZoneAirCooled_SingleAgent(Building_5ZoneAirCooled):
    def __init__(self, args = None):
        #
        super().__init__(args)
        #
        # the pairing which gives, which agent (identified by name) controlles which device (or zone) and what kind of device it is
        self.agent_device_pairing = {"MainAgent":
                                         ("all", "5ZoneAirCooled,SingleAgent,RL")
                                     }

    def obtain_cobs_actions(self, agent_actions, next_timestep):
        """
        Returns a list of actions that can be passed to cobs.

        The agent_actions dict is expected to have the format:
        {agent_name: {controlled_parameter: new_value}}
        """
        actions = []
        for agent_name, ag_actions in agent_actions.items():
            for zone in self.room_names:
                damper_pos_val = ag_actions[f"{zone} Zone VAV Reheat Damper Position"]
                actions.append({"priority": 0,
                                "component_type":"Schedule:Constant",
                                "control_type":  "Schedule Value",
                                "actuator_key": f"{zone} VAV Customized Schedule",
                                "value": damper_pos_val,
                                "start_time": next_timestep})
                mean_temp_sp = ag_actions[f"{zone} Zone Heating/Cooling-Mean Setpoint"]
                delta = ag_actions[f"{zone} Zone Heating/Cooling-Delta Setpoint"]
                if delta < 0.1: delta = 0.1
                actions.append({"value":      mean_temp_sp - delta,
                                "start_time": next_timestep,
                                "priority":   0,
                                "component_type": "Zone Temperature Control",
                                "control_type":   "Heating Setpoint",
                                "actuator_key":   zone})
                actions.append({"value":      mean_temp_sp + delta,
                                "start_time": next_timestep,
                                "priority":   0,
                                "component_type": "Zone Temperature Control",
                                "control_type":   "Cooling Setpoint",
                                "actuator_key":   zone})
        return actions

class Building_5ZoneAirCooled_SmallSingleAgent(Building_5ZoneAirCooled):
    def __init__(self, args = None):
        #
        super().__init__(args)
        #
        # the pairing which gives, which agent (identified by name) controlles which device (or zone) and what kind of device it is
        self.agent_device_pairing = {"MainAgent":
                                         ("all", "5ZoneAirCooled,SingleAgent,RL,VerySmall")
                                     }

    def obtain_cobs_actions(self, agent_actions, next_timestep):
        """
        Returns a list of actions that can be passed to cobs.

        The agent_actions dict is expected to have the format:
        {agent_name: {controlled_parameter: new_value}}
        """
        actions = []
        for agent_name, ag_actions in agent_actions.items():
            for zone in self.room_names:
                # set damper position to 1
                damper_pos_val = 1
                actions.append({"priority": 0,
                                "component_type":"Schedule:Constant",
                                "control_type":  "Schedule Value",
                                "actuator_key": f"{zone} VAV Customized Schedule",
                                "value": damper_pos_val,
                                "start_time": next_timestep})
                mean_temp_sp = ag_actions[f"{zone} Zone Heating/Cooling-Mean Setpoint"]
                delta = ag_actions[f"{zone} Zone Heating/Cooling-Delta Setpoint"]
                if delta < 0.1: delta = 0.1
                actions.append({"value":      mean_temp_sp - delta,
                                "start_time": next_timestep,
                                "priority":   0,
                                "component_type": "Zone Temperature Control",
                                "control_type":   "Heating Setpoint",
                                "actuator_key":   zone})
                actions.append({"value":      mean_temp_sp + delta,
                                "start_time": next_timestep,
                                "priority":   0,
                                "component_type": "Zone Temperature Control",
                                "control_type":   "Cooling Setpoint",
                                "actuator_key":   zone})
        return actions



