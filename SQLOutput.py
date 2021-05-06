
import sqlite3

class SQLOutput():
    
    def __init__(self, filename, building):
        """
        Initializes the object for outputtng things to a SQLite Database or another
        database.
        """
        self.db = sqlite3.connect(filename)
        self.building = building
        #
        # all output dictionaries have the same format:
        # database table name: [database column type, local variable name in code]
        #

        # output vars every episode at the end accumulated or somehow aggreated
        self.output_vars_eels = {
            "episode":  ["integer", "episode"],
            "lr":       ["integer", "lr"],
            "tau":      ["float",   "tau"],
            "ou_theta": ["float",   "ou_theta"],
            "ou_sigma": ["float",   "ou_sigma"],
            "ou_mu":    ["float",   "ou_mu"],
            "lambda_energy":  ["float","lambda_energy"],
            "lambda_manu_stp":["float","lambda_manu_stp"],
            "epsilon":  ["float",   "epsilon"],
            "time_cons":["float",   "t_diff"],
            "target_netw_u":  ["boolean", "target_network_update"],
            "eval_epoch":     ["boolean", "evaluation_epoch"],
            "random_process_addition": ["boolean","random_process_addition"],
            "sum_manual_stp_ch_n":  ["float", "sum_manual_stp_ch_n"],
            "mean_manual_stp_ch_n": ["float", "mean_manual_stp_ch_n"],
            "mean_reward": ["float", "reward_mean"],
            "sum_reward":  ["float", "reward_sum"],
            "mean_energy_Wh":   ["float", "current_energy_Wh_mean"],
            "sum_energy_Wh":    ["float", "current_energy_Wh_sum"],
            # informations about the agents
            "loss_mean": ["float",   "loss_mean"],
            "q_st2_mean":["float",   "q_st2_mean"],
            "J_mean":    ["float",   "J_mean"],
            "frobnorm_agent_matr_mean": ["float", "frobnorm_agent_matr_mean"],
            "frobnorm_agent_bias_mean": ["float", "frobnorm_agent_bias_mean"],
            "frobnorm_critic_matr_mean":["float", "frobnorm_critic_matr_mean"],
            "frobnorm_critic_bias_mean":["float", "frobnorm_critic_bias_mean"]
        }

        # output vars after some episodes, but every step
        self.output_vars_sees = {
            "episode":  ["integer", "episode_number"],
            "step":     ["integer", "timestep"],
            "datetime": ["string",  "currdate"],
            "reward":   ["float",   "reward"],
            "manual_stp_ch_n":  ["float", "n_manual_stp_changes"],
            "energy_Wh":        ["float", "current_energy_Wh"],
            # these can be found in the state dict
            "outdoor_temp":       ["float", "Outdoor Air Temperature"],
            "outdoor_humidity":   ["float", "Outdoor Air Humidity"],
            "outdoor_windspeed":  ["float", "Outdoor Wind Speed"],
            "outdoor_winddir":    ["float", "Outdoor Wind Direction"],
            "outdoor_solar_radi_dir":   ["float", "Outdoor Solar Radi Direct"],
            "outdoor_solar_radi_indir": ["float", "Outdoor Solar Radi Diffuse"]
        }

        # output vars after some episodes, but every step, and every room
        self.output_vars_sees_er = {
            "episode":  ["integer", "episode_number"],
            "step":     ["integer", "timestep"],
            "room":     ["string",  None],
            "temp":     ["float",   "Zone Temperature"],
            "occupancy":["integer", "Zone People Count"],
            "co2":      ["float",   "Zone CO2"],
            "humidity": ["float",   "Zone Relative Humidity"],
            "target_temp": ["float", "target_temp_per_room"]
        }

        # output vars after some episodes, but every step, every agent
        self.output_vars_seesea = {
            "episode":  ["integer", "episode_number"],
            "step":     ["integer", "timestep"],
            "agent_nr": ["integer", None],
            "agent_actions": ["string", None]
        }


    def initialize(self):
        """
        Initializes the database and this object
        """
        # initialize the database
        for table in ["eels", "sees", "seesea", "sees_er"]:
            self.db.execute( self.get_sql_create_statement(table) )


    def get_sql_create_statement(self, vartype):
        """
        Returns a SQL CREATE TABLE string for creating the table for `vartype`.
        `vartype` should be out of the list: [eels, sees, seesea, sees_er]
        """
        outputstr = f"CREATE TABLE {vartype}("
        vardict = {"eels":   self.output_vars_eels,
                   "sees":   self.output_vars_sees,
                   "seesea": self.output_vars_seesea,
                   "sees_er":self.output_vars_sees_er}[vartype]
        for key, val in vardict.items():
            outputstr += f"{key} {val[0]},"
        return outputstr[:-1] + ");"


    def _propagate_to_db(self, tablename, dict_for_db):
        colstr, valstr = "", ""
        for col, val in dict_for_db.items():
            colstr += f"{col},"
            if type(valstr) == str:
                valstr += f"\"{val}\","
            else:
                valstr += f"{val},"
        self.db.execute(f"INSERT INTO {tablename}({colstr[:-1]}) VALUES({valstr[:-1]});")


    # if ignore_agents is set to True, it will ignore the agent values
    def add_every_step_of_episode(self, local_vars, ignore_agents = False):
        # get global data
        dict_for_db = {}
        for colname, (_, lvarname) in self.output_vars_eees.items():
            if not lvarname is None:
                dict_for_db[colname] = local_vars[lvarname]
        self._propagate_to_db("eees", dict_for_db)
        #
        if ignore_agents:
            return
        # actions for every agent
        for idx, agent in enumerate(local_vars["agents"]):
            dict_for_db = {}
            for colname, (_, lvarname) in self.output_vars_eeesea.items():
                if lvarname is None:
                    lvarval = idx
                elif colname == "episode" or colname == "step":
                    lvarval = local_vars[lvarname]
                else:
                    lvarval = local_vars[lvarname][idx]
                dict_for_db[colname] = lvarval
            self._propagate_to_db("eeesea", dict_for_db)
            # break loop, if there is no 
            if agent.shared_network_per_agent_class:
                break


    def add_last_step_of_episode(self, local_vars):
        dict_for_db = {}
        for colname, (_, lvarname) in self.output_vars_eels.items():
            if not lvarname is None and lvarname in local_vars.keys():
                dict_for_db[colname] = local_vars[lvarname]
            else:
                dict_for_db[colname] = 0
        self._propagate_to_db("eels", dict_for_db)


    def add_every_step_of_some_episodes(self, local_vars):
        # get global data
        dict_for_db = {}
        for colname, (_, lvarname) in self.output_vars_sees.items():
            if lvarname is None:
                continue
            if lvarname in local_vars["state"].keys():
                dict_for_db[colname] = local_vars["state"][lvarname]
            else:
                dict_for_db[colname] = local_vars[lvarname]
        self._propagate_to_db("sees", dict_for_db)
        # get data for every room
        for idx, room in enumerate(self.building.room_names):
            dict_for_db = {}
            for colname, (_, lvarname) in self.output_vars_sees_er.items():
                if lvarname is None:
                    if colname == "room":
                        dict_for_db[colname] = room
                    else:
                        continue
                elif colname == "episode" or colname == "step":
                    dict_for_db[colname] = local_vars[lvarname]
                elif colname == "target_temp":
                    if room in local_vars["target_temp_per_room"].keys():
                        dict_for_db[colname] = local_vars["target_temp_per_room"][room]
                    else:
                        dict_for_db[colname] = 0
                else:
                    dict_for_db[colname] = local_vars["state"][f"{room} {lvarname}"]
            self._propagate_to_db("sees_er", dict_for_db)
        # get data for every agent
        for idx, agent_name in enumerate(self.building.agent_device_pairing.keys()):
            dict_for_db = {
                "episode":       local_vars["episode_number"],
                "step":          local_vars["timestep"],
                "agent_nr":      idx,
                "agent_actions": local_vars["agent_actions_dict"][agent_name]
            }
            self._propagate_to_db("seesea", dict_for_db)


