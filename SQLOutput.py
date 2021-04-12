
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
        # output vars every episode, every step
        self.output_vars_eees = {
            "episode": ["integer", "episode_number"],
            "step":    ["integer", "timestep"],
            "reward":  ["float",   "reward"],
            "manual_stp_ch_n":   ["float", "n_manual_stp_changes"],
            #"manual_stp_ch_mag": ["integer", "mag_manual_stp_changes"],
            "energy_Wh":  ["float",   "current_energy_Wh"]
        }

        # output vars every episode, every step, every agent
        self.output_vars_eeesea = {
            "episode":  ["integer", "episode_number"],
            "step":     ["integer", "timestep"],
            "agent_nr": ["integer", None],
            "loss":     ["float",   "output_loss_list"],
            "q_st2":    ["float",   "output_q_st2_list"],
            "J":        ["float",   "output_J_mean_list"],
            "frobnorm_agent_matr": ["float", "output_ag_frobnorm_mat_list"],
            "frobnorm_agent_bias": ["float", "output_ag_frobnorm_bia_list"],
            "frobnorm_critic_matr":["float", "output_cr_frobnorm_mat_list"],
            "frobnorm_critic_bias":["float", "output_cr_frobnorm_bia_list"]
        }

        # output vars every episode at the end accumulated or somehow aggreated
        self.output_vars_eels = {
            "episode":  ["integer", "episode_number"],
            "lr":       ["integer", "lr"],
            "tau":      ["float",   "tau"],
            "ou_theta": ["float",   None],
            "ou_sigma": ["float",   None],
            "ou_mu":    ["float",   None],
            "lambda_energy":  ["float","lambda_energy"],
            "lambda_manu_stp":["float","lambda_manu_stp"],
            "epsilon":  ["float",   "epsilon"],
            "time_cons":["float",   "t_diff"],
            "target_netw_u":  ["boolean", "target_network_update"],
            "eval_epoch":     ["boolean", "evaluation_epoch"],
            "random_process_addition": ["boolean","random_process_addition"]
        }

        # output vars after some episodes, but every step
        self.output_vars_sees = {
            "episode":  ["integer", "episode_number"],
            "step":     ["integer", "timestep"],
            "datetime": ["string",  "currdate"],
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
            "room":     ["integer", None],
            "temp":     ["float",   "Zone Temperature"],
            "occupancy":["integer", "Zone People Count"],
            "co2":      ["float",   "Zone CO2"],
            "humidity": ["float",   "Zone Relative Humidity"]
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
        for table in ["eees", "eeesea", "eels", "sees", "seesea", "sees_er"]:
            self.db.execute( self.get_sql_create_statement(table) )


    def get_sql_create_statement(self, vartype):
        """
        Returns a SQL CREATE TABLE string for creating the table for `vartype`.
        `vartype` should be out of the list: [eees, eeesea, eels, sees, seesea, sees_er]
        """
        outputstr = f"CREATE TABLE {vartype}("
        vardict = {"eees":   self.output_vars_eees,
                   "eeesea": self.output_vars_eeesea,
                   "eels":   self.output_vars_eels,
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
                        dict_for_db[colname] = idx
                    else:
                        continue
                elif colname == "episode" or colname == "step":
                    dict_for_db[colname] = local_vars[lvarname]
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


