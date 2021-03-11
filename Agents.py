
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

        input_size  = len(self.input_parameters)
        output_size = len(self.controlled_parameters)
        hidden_size = (2*input_size+output_size) // 3

        self.model_actor = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size)
        )
        self.model_target = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size)
        )
        # copy weights from actor -> target
        for mactor_param, mtarget_param in zip(self.model_actor.parameters(), self.model_target.parameters()):
            mtarget_param.data.copy_(mactor_param.data)

        # initialize the optimizer
        self.optimizer = torch.optim.Adam(params = self.model_actor.parameters(), lr = self.lr)

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
            ou_sample      = self.ou_process.sample()
            ou_sample      = ou_sample.astype(np.float32) # we need this line for torch 1.2.0, torch 1.8.0 does not need this any more
            output_tensor += torch.from_numpy(ou_sample[np.newaxis, :])
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
        torch.save(self.model_actor,  os.path.join(storage_dir, prefix + self.name + "_model_actor.pickle"))
        torch.save(self.model_target, os.path.join(storage_dir, prefix + self.name + "_model_target.pickle"))

    def load_models_from_disk(self, storage_dir, prefix=""):
        self.model_actor = torch.load(os.path.join(storage_dir, prefix + self.name + "_model_actor.pickle"))
        self.model_target= torch.load(os.path.join(storage_dir, prefix + self.name + "_model_target.pickle"))




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

    def step(self, current_state):
        output_dic = {}
        # ... do not do anything by default
        #output_dic = {"VAV Reheat Damper Position": 0.8,
        #              "Zone Heating/Cooling-Mean Setpoint": 22.0,
        #              "Zone Heating/Cooling-Delta Setpoint": 2.0}
        if current_state["Zone Relative Humidity"] > 0.7 or \
           current_state["Zone CO2"] > 0.2332: # 1500 ppm
            output_dic["VAV Reheat Damper Position"] = 1
        if current_state["Zone Temperature"] > 0.5: # 25 degree C
            output_dic["Zone Heating/Cooling-Mean Setpoint"] = 0.4
            output_dic["Zone Heating/Cooling-Delta Setpoint"] = 0.1
        elif current_state["Zone Temperature"] < 0.3: # 19 degree C
            output_dic["Zone Heating/Cooling-Mean Setpoint"] = 0.4
            output_dic["Zone Heating/Cooling-Delta Setpoint"] = 0.1
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


