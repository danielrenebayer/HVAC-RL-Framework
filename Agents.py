
import os
import torch
import numpy as np

from RandomProcessExt import OrnsteinUhlenbeckProcess

def agent_constructor(zone_class, rl_storage_filepath=None):
    if zone_class == "VAV with Reheat,Heating,Cooling,NoRL":
        new_agent = AgentNoRL_VAVRhHC()
        new_agent.input_parameters = [
            "Outdoor Air Temperature",
            "Outdoor Air Humidity",
            "Outdoor Wind Speed",
            'Outdoor Wind Direction',
            'Outdoor Solar Radi Diffuse',
            'Outdoor Solar Radi Direct',
            "Zone Relative Humidity",
            "Zone VAV Reheat Damper Position",
            "Zone CO2",
            "Zone People Count",
            "Zone Temperature"]
        new_agent.controlled_parameters = ["Zone VAV Reheat Damper Position", "Zone Heating/Cooling-Mean Setpoint", "Zone Heating/Cooling-Delta Setpoint"]

    elif zone_class == "VAV with Reheat,Heating,Cooling,RL":
        new_agent = AgentRL(zone_class)
        new_agent.input_parameters = [
            "Minutes of Day",
            "Day of Week",
            "Calendar Week",
            "Outdoor Air Temperature",
            "Outdoor Air Humidity",
            "Outdoor Wind Speed",
            'Outdoor Wind Direction',
            'Outdoor Solar Radi Diffuse',
            'Outdoor Solar Radi Direct',
            "Zone Relative Humidity",
            "Zone VAV Reheat Damper Position",
            "Zone CO2",
            "Zone People Count",
            "Zone Temperature"]
        new_agent.controlled_parameters = ["Zone VAV Reheat Damper Position", "Zone Heating/Cooling-Mean Setpoint", "Zone Heating/Cooling-Delta Setpoint"]

    elif zone_class == "5ZoneAirCooled,SingleAgent,RL":
        new_agent = AgentRL(zone_class)
        new_agent.input_parameters = [
            "Minutes of Day",
            "Day of Week",
            "Calendar Week",
            "Outdoor Air Temperature",
            "Outdoor Air Humidity",
            "Outdoor Wind Speed",
            'Outdoor Wind Direction',
            'Outdoor Solar Radi Diffuse',
            'Outdoor Solar Radi Direct']
        for zone in [f"SPACE{i}-1" for i in range(1,6)]:
            new_agent.input_parameters.extend([
                f"{zone} Zone Relative Humidity",
                f"{zone} Zone VAV Reheat Damper Position",
                f"{zone} Zone CO2",
                f"{zone} Zone People Count",
                f"{zone} Zone Temperature"])
        new_agent.controlled_parameters = []
        for zone in [f"SPACE{i}-1" for i in range(1,6)]:
            new_agent.controlled_parameters.extend([
                f"{zone} Zone VAV Reheat Damper Position",
                f"{zone} Zone Heating/Cooling-Mean Setpoint",
                f"{zone} Zone Heating/Cooling-Delta Setpoint"])

    elif zone_class == "5ZoneAirCooled,SingleAgent,RL,VerySmall":
        new_agent = AgentRL(zone_class)
        new_agent.input_parameters = [
            "Minutes of Day",
            "Day of Week"]
        for zone in [f"SPACE{i}-1" for i in range(1,6)]:
            new_agent.input_parameters.extend([
                f"{zone} Zone People Count"])
        new_agent.controlled_parameters = []
        for zone in [f"SPACE{i}-1" for i in range(1,6)]:
            new_agent.controlled_parameters.extend([
                f"{zone} Zone Heating/Cooling-Mean Setpoint",
                f"{zone} Zone Heating/Cooling-Delta Setpoint"])

    elif zone_class == "VAV with Reheat,Heating,Cooling,RL,VerySmall":
        new_agent = AgentRL(zone_class)
        new_agent.input_parameters = [
            "Minutes of Day",
            "Day of Week",
            #"Outdoor Air Temperature",
            #'Outdoor Solar Radi Direct',
            "Zone People Count"]
        new_agent.controlled_parameters = [
                "Zone Heating/Cooling-Mean Setpoint",
                "Zone Heating/Cooling-Delta Setpoint"]

    else:
        raise AttributeError(f"Unknown zone class: {zone_class}")
    return new_agent




class AgentRL:
    
    def __init__(self, class_name):
        self.class_name = class_name
        self.input_parameters = []
        self.controlled_parameters = []
        self.model_actor = None
        self.model_target= None
        self.trafo_matrix= None
        self.initialized = False
        self.name = ""
        self.controlled_element = ""
        self.optimizer = None
        self.ou_process= None

    def initialize(self, name, controlled_element, global_state_keys, args = None):
        """
        Initializes the agent.
        This function should be callen after the creation of the object and the
        corret set of the input / output variables.

        Parameters
        ----------
        name : str
            The name of the agent.
        controlled_element : str
            The name of the element (as uniquly named in the building), that is controlled.
        global_state_keys : list of str
            The list of all keys (in the correct order, as they occur in the input tensor later!) in the state.
        args : argparse
            Additional arguments, optional.
        """
        if len(self.input_parameters) == 0 or len(self.controlled_parameters) == 0:
            raise RuntimeError("The number of input and output parameters has to be greater than 0")
        self.initialized = True
        self.name = name
        self.controlled_element = controlled_element
        self.use_cuda = torch.cuda.is_available() if args is None else args.use_cuda
        self.lr     = 0.001 if args is None else args.lr
        self.ou_theta = 0.3 if args is None else args.ou_theta
        self.ou_mu    = 0.0 if args is None else args.ou_mu
        self.ou_sigma = 0.3 if args is None else args.ou_sigma
        self.w_l2     = 0.00001 if args is None else args.agent_w_l2
        self.ou_update_freq = 1 if args is None else args.ou_update_freq
        self._stepts_since_last_ou_update = self.ou_update_freq
        self._last_ou_sample = None

        input_size  = len(self.input_parameters)
        output_size = len(self.controlled_parameters)
        hidden_size = (2*input_size+output_size) // 3

        self.model_actor = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.LeakyReLU(),
            #torch.nn.Linear(hidden_size, hidden_size),
            #torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_size, output_size),
            torch.nn.Tanh()
        )
        self.model_target = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.LeakyReLU(),
            #torch.nn.Linear(hidden_size, hidden_size),
            #torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_size, output_size),
            torch.nn.Tanh()
        )
        # change initialization
        for m_param in self.model_actor.parameters():
            if len(m_param.shape) == 1:
                # other initialization for biases
                torch.nn.init.constant_(m_param, 0.001)
            else:
                torch.nn.init.normal_(m_param, 0.0, 1.0)
        # copy weights from actor -> target
        for mactor_param, mtarget_param in zip(self.model_actor.parameters(), self.model_target.parameters()):
            mtarget_param.data.copy_(mactor_param.data)

        # initialize the optimizer
        self.optimizer = torch.optim.Adam(params = self.model_actor.parameters(), lr = self.lr, weight_decay = self.w_l2)

        # initialize the OU-Process
        self.ou_process = OrnsteinUhlenbeckProcess(theta = self.ou_theta,
                                                   mu    = self.ou_mu,
                                                   sigma = self.ou_sigma,
                                                   size  = output_size)

        # define the transformation matrix
        trafo_list = []
        for input_param in self.input_parameters:
            try:
                idx = global_state_keys.index(input_param) # this may rise a ValueError
            except ValueError:
                # if a ValueError occurs, the element is not in the list
                # this means, that we have to add the name of the controlled device
                # in front of the input parameter to find the element
                idx = global_state_keys.index(f"{self.controlled_element} {input_param}")
            row = torch.zeros(len(global_state_keys))
            row[idx] = 1.0
            trafo_list.append(row)
        self.trafo_matrix = torch.stack(trafo_list).T

        # Move things to GPU, if selected
        if self.use_cuda:
            self.model_actor = self.model_actor.to(0)
            self.model_target= self.model_target.to(0)
            self.trafo_matrix = self.trafo_matrix.to(0)

    def optimizer_step(self):
        """
        Applies on step by the optimizer.
        """
        self.optimizer.step()

    def prepare_state_dict(self, current_state):
        """
        Translates the current state (or a list of states) as dict to a pytorch-tensor, that can be passed to RLAgent.step_tensor().
        """
        if type(current_state) == dict:
            input_vec = np.zeros( len(self.input_parameters) )
            for idx, ip in enumerate(self.input_parameters):
                input_vec[idx] = current_state[ip]
            return torch.from_numpy(input_vec[np.newaxis, :].astype(np.float32))

        elif type(current_state) == list:
            raise NotImplemented("This function should be implemented to be used.")

    def step_tensor(self, state_tensor, use_actor = True, add_ou = False):
        """
        Computes the next step and outputs the resulting torch tensor.

        Parameters
        ---------
        state_tensor : torch.tensor
            The global state tensor.
        use_actor : Boolean
            If set to true, it will use the actor model for infering the next state, otherwise
            it will use the target network.
            Defaults to True.
        add_ou : Boolean
            If set to true, it will add the result of the OU-process to the output.
            Defaults to False.
        """
        if type(state_tensor) == list:
            state_tensor = torch.cat(state_tensor, dim=0)
        if self.use_cuda:
            state_tensor = state_tensor.to(0)
        input_tensor = torch.matmul(state_tensor, self.trafo_matrix)
        if use_actor:
            output_tensor  = self.model_actor(input_tensor)
        else:
            output_tensor  = self.model_target(input_tensor)
        if self.use_cuda:
            output_tensor  = output_tensor.cpu()
        if add_ou:
            if self._stepts_since_last_ou_update >= self.ou_update_freq:
                self._stepts_since_last_ou_update = 1
                ou_sample  = self.ou_process.sample()
                self._last_ou_sample = ou_sample
            else:
                self._stepts_since_last_ou_update += 1
            ou_sample      = self._last_ou_sample.astype(np.float32) # we need this line for torch 1.2.0, torch 1.8.0 does not need this any more
            output_tensor += torch.from_numpy(ou_sample[np.newaxis, :])
            output_tensor  = torch.clamp(output_tensor, -1.0, 1.0)
        return output_tensor

    def output_tensor_to_action_dict(self, output_tensor):
        np_tensor = output_tensor.detach().numpy()
        if len(np_tensor.shape) > 1:
            # in this case we have more batches; we only want the first one
            np_tensor = np_tensor[0, :]
        output_dict = {}
        for idx, cp in enumerate(self.controlled_parameters):
            output_dict[cp] = np_tensor[idx]
        return output_dict

    def action_dict_to_output_tensor(self, action_dict):
        output_tensor = np.zeros(len(self.controlled_parameters))
        for idx, cp in enumerate(self.controlled_parameters):
            output_tensor[idx] = action_dict[cp]
        return torch.tensor(output_tensor[np.newaxis, :].astype(np.float32))

    def step(self, current_state, use_actor = True, add_ou = False):
        """
        Computes the next step. Outputs the resulting torch tensor and the tensor as converted dict.
        
        Parameters
        ---------
        current_state : dict
            the normalized (!) current current
        use_actor : Boolean
            If set to true, it will use the actor model for infering the next state, otherwise
            it will use the target network.
            Defaults to True.
        
        """
        if not self.initialized:
            raise RuntimeError("Agent is not initialized.")
        #
        # first: select the needed input parameters and sort them in the right direction
        input_ten = self.prepare_state_dict(current_state)
        #
        # apply function approximator (i.e. neural network)
        output_ten = self.step_tensor(input_ten, use_actor, add_ou)
        #
        # output as dict
        output_dic = self.output_tensor_to_action_dict(output_ten)
        return output_ten, output_dic, input_ten
    
    def update_target_network(self, tau = 0.01):
        """
        Propagates the parameters of the actor network to those of the target network.
        """
        for mactor_param, mtarget_param in zip(self.model_actor.parameters(), self.model_target.parameters()):
            mtarget_param.data.copy_( (1-tau) * mtarget_param.data + tau * mactor_param.data)

    def save_models_to_disk(self, storage_dir, prefix=""):
        torch.save(self.model_actor,  os.path.join(storage_dir, prefix + "_model_actor.pickle"))
        torch.save(self.model_target, os.path.join(storage_dir, prefix + "_model_target.pickle"))

    def load_models_from_disk(self, storage_dir, prefix=""):
        self.model_actor = torch.load(os.path.join(storage_dir, prefix + "_model_actor.pickle"))
        self.model_target= torch.load(os.path.join(storage_dir, prefix + "_model_target.pickle"))




class AgentNoRL:
    
    def __init__(self, class_name):
        self.class_name = class_name
        self.input_parameters = []
        self.controlled_parameters = []
        self.ctrl_fx = None

    def step(self, current_state):
        #
        # first: select the needed input parameters and sort them in the right direction
        input_vec = np.zeros( len(self.input_parameters) )
        output_vec = np.zeros( len(self.controlled_parameters) )
        for idx, ip in enumerate(self.input_parameters):
            input_vec[idx] = current_state[ip]
        #
        # apply control function
        if not self.ctrl_fx is None:
            output_vec = self.ctrl_fx(input_vec)
        #
        # output as dict
        output_dic = {}
        for idx, cp in enumerate(self.controlled_parameters):
            output_dic[cp] = output_vec[idx]
        return output_dic



class AgentNoRL_VAVRhHC(AgentNoRL):
    
    def __init__(self):
        super().__init__("VAV with Reheat,Heating,Cooling,NoRL")

    def initialize(self, name, controlled_element):
        self.name = name
        self.controlled_element = controlled_element

    def step(self, current_state):
        output_dic = {"Zone VAV Reheat Damper Position":    0.75,
                      "Zone Heating/Cooling-Mean Setpoint": 23.0,
                      "Zone Heating/Cooling-Delta Setpoint": 7.0}
        # set damper position to maximal value if humidity or co2 is high
        if current_state[f"{self.controlled_element} Zone Relative Humidity"] > 0.8 or \
           current_state[f"{self.controlled_element} Zone CO2"] > 1500: # ppm
               output_dic["Zone VAV Reheat Damper Position"] = 1.0
        #
        # rule based setting of the heating/cooling setpoint and delta
        #  Mo - Fr, 7.00 - 18.00 h: 21.5 deg C, Delta 1 deg C
        if current_state["time"].weekday() <= 4:
            if current_state["time"].hour >= 7 and current_state["time"].hour <= 18:
                output_dic["Zone Heating/Cooling-Mean Setpoint"] = 21.5
                output_dic["Zone Heating/Cooling-Delta Setpoint"] = 1.0
                if output_dic["Zone VAV Reheat Damper Position"] < 0.9:
                    output_dic["Zone VAV Reheat Damper Position"] = 0.9
        return output_dic
    
    
    




class OneAgentOneDevice:
    
    
    def __init__(self, device_class, rl_storage_filepath=None):
        """
        Constructs a controlling agent for a (single) existing device.
        
        Parameters
        ----------
        device_class : str
            The name of the class of the device, e.g. 'VAV Reheat', 'Central Boiler' or 'Central Chiller'
        rl_storage_filepath : str
            Defaults to None.
            The path to the file containing the serialized object of the already trained RL-agent
            If this parameter is not set, it will use a randomly initialized RL-agent
        """
        if device_class == "Central Boiler":
            self.input_parameters = ["Outdoor Temp", "Outdoor Hum", "..."]
            self.controlled_parameters = ["setpoint"]





class OneAgentOneZone:
    
    
    def __init__(self, zone_class, rl_storage_filepath=None):
        """
        Constructs a controlling agent for a zone.
        
        Parameters
        ----------
        device_class : str
            The name of the class of the device, e.g. 'VAV Reheat', 'Central Boiler' or 'Central Chiller'
        rl_storage_filepath : str
            Defaults to None.
            The path to the file containing the serialized object of the already trained RL-agent
            If this parameter is not set, it will use a randomly initialized RL-agent
        """
        if zone_class == "VAV with Reheat,Heating,Cooling,NoRL":
            self.input_parameters = [
                "Outdoor Air Temperature",
                "Outdoor Air Humidity",
                "Outdoor Wind Speed",
                'Outdoor Wind Direction',
                'Outdoor Solar Radi Diffuse',
                'Outdoor Solar Radi Direct',
                "Zone Relative Humidity",
                "Zone VAV",
                "Zone CO2"
                "Zone Temperature"]
            self.controlled_parameters = ["VAV Setpoint", "Zone Heating Setpoint", "Zone Cooling Setpoint"]
        else:
            raise AttributeError(f"Unknown zone class: {zone_class}")


