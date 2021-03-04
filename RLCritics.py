
#
# One can find all critics, that are needed for DDPG
#

import torch
import numpy as np

class CriticMergeAndOnlyFC:
    """
    This critic is a neural network with 3 fully connected layer.
    The state and the action vector are merged.
    """
    
    def __init__(self, input_variables, agents, hidden_size):
        self.input_variables = input_variables
        self.agent_variables = [ f"{agent.name} {controlled_var}" for agent in agents for controlled_var in agent.controlled_parameters ]
        self.agent_var_sorted= [ (agent.name, agent.controlled_parameters) for agent in agents ]
        # Hint: agent_var_sorted is the dict (actually it is a list so prohibit reordering) that is used by self.forward()
        # agent_variables is only for the publicity to know that to do
        #
        self.input_state_var_size = len(input_variables)
        self.input_agent_var_size = len(self.agent_variables)
        self.input_size  = self.input_state_var_size + self.input_agent_var_size
        self.hidden_size = hidden_size
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 1),
            torch.nn.ReLU()
        )
        self.model_target = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 1),
            torch.nn.ReLU()
        )
        # copy weights from Q -> target Q
        for m_param, mtarget_param in zip(self.model.parameters(), self.model_target.parameters()):
            mtarget_param.data.copy_(m_param.data)


    def prepare_state_dict(self, current_state):
        """
        Translates the current state (or a list of states) as dict to a pytorch-tensor, that can be passed to CriticMergeAndOnlyFC.forward_tensor().
        """
        if type(current_state) == dict:
            input_vec = np.zeros( self.input_state_var_size )
            for idx, ip in enumerate(self.input_variables):
                input_vec[idx] = current_state[ip]
            return torch.from_numpy(input_vec[np.newaxis, :].astype(np.float32))

        elif type(current_state) == list:
            batch_list = []
            for state in current_state:
                input_vec = np.zeros( self.input_state_var_size )
                for idx, ip in enumerate(self.input_variables):
                    input_vec[idx] = current_state[ip]
                batch_list.append(input_vec)
            return torch.from_numpy(np.array(batch_list).astype(np.float32))


    def forward_tensor(self, state_tensor, all_actions_tensor, no_target = True):
        """
        Computes the next step. Outputs the resulting torch tensor.
        In contrast to forward() it accepts torch tensors as input.
        
        For parameter reference see forward()
        """
        # cat state tensor and action tensor
        input_ten = torch.cat([state_tensor, all_actions_tensor], dim=1)
        if no_target:
            return self.model(input_ten)
        else:
            return self.model_target(input_ten)


    def prepare_action_dict(self, all_actions):
        """
        Translates the actions of the agents as dict to a pytorch-tensor, that can be passed to CriticMergeAndOnlyFC.forward_tensor().
        """
        idx = 0
        input_vec = np.zeros(self.input_agent_var_size)
        for agent_name, param_list in self.agent_var_sorted:
            for param_name in param_list:
                input_vec[idx] = all_actions[agent_name][param_name]
                idx += 1
        return torch.from_numpy(input_vec[np.newaxis, :].astype(np.float32))


    def forward(self, current_state, all_actions, no_target = True):
        """
        Computes the next step. Outputs the resulting torch tensor and the tensor as numpy array.
        
        Parameters
        ---------
        current_state : dict
            the normalized (!) current current
        all_actions : dict
            the normalized (!) actions as a dict of the form {agent_name: {param_name: value}}
        no_target : Boolean
            If set to true, it will use the critic model for infering the next state, otherwise
            it will use the target network.
            Defaults to True.
        """
        #
        # first: select the needed input variables and sort them in the right direction
        input1_ten = self.prepare_state_dict(current_state)
        #
        # second: add all actions in the correct order to the input_vec
        input2_ten = self.prepare_action_dict(all_actions)
        #
        # apply function approximator (i.e. neural network)
        input_ten = torch.cat([input1_ten, input2_ten], dim=1)
        if no_target:
            output_ten = self.model(input_ten)
        else:
            output_ten = self.model_target(input_ten)
        return output_ten, output_ten.detach().numpy()[0:, ]


    def update_target_network(self, tau = 0.01):
        """
        Propagates the parameters of the critic network to those of the target network.
        """
        for m_param, mtarget_param in zip(self.model.parameters(), self.model_target.parameters()):
            mtarget_param.data.copy_( (1-tau) * mtarget_param.data + tau * m_param.data)

