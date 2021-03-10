"""
This file contains the default buildings, that can be used in EnergyPlus using the COBS API
"""

import os
import cobs
from global_paths import global_paths

class Building_5ZoneAirCooled:
    def __init__(self):
        #
        #
        self.room_names = [f'SPACE{i}-1' for i in range(1,6)]
        #
        # the pairing which gives, which agent (identified by name) controlles which device (or zone) and what kind of device it is
        self.agent_device_pairing = {f"Agent SPACE{i}-1":
                                         (f"SPACE{i}-1", "VAV with Reheat,Heating,Cooling,RL")
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
        #
        # define the model
        model = cobs.Model(
            idf_file_name = os.path.join(global_paths["COBS"], "cobs/data/buildings/5ZoneAirCooled.idf"),
            weather_file  = os.path.join(global_paths["COBS"], "cobs/data/weathers/USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw"),
            #weather_file  = os.path.join(global_paths["COBS"], "cobs/data/weathers/1.epw"),
            eplus_naming_dict = self.eplus_extra_states,
            eplus_var_types   = self.eplus_var_types
        )
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


class Building_5ZoneAirCooled_SingleAgent:
    def __init__(self):
        #
        #
        self.room_names = [f'SPACE{i}-1' for i in range(1,6)]
        #
        # the pairing which gives, which agent (identified by name) controlles which device (or zone) and what kind of device it is
        self.agent_device_pairing = {"MainAgent":
                                         ("all", "5ZoneAirCooled,SingleAgent,RL")
                                     }
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
        #
        # define the model
        model = cobs.Model(
            idf_file_name = os.path.join(global_paths["COBS"], "cobs/data/buildings/5ZoneAirCooled.idf"),
            weather_file  = os.path.join(global_paths["COBS"], "cobs/data/weathers/USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw"),
            #weather_file  = os.path.join(global_paths["COBS"], "cobs/data/weathers/1.epw"),
            eplus_naming_dict = self.eplus_extra_states,
            eplus_var_types   = self.eplus_var_types
        )
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


