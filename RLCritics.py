
#
# One can find all critics, that are needed for DDPG
#

import os
import torch
import numpy as np

class CriticMergeAndOnlyFC:
    """
    This critic is a neural network with 3 fully connected layer.
    The state and the action vector are merged.
    """
    
    def __init__(self, input_variables, agents, global_state_keys, args=None):
        """
        Initializes a critic, that contains a fully-connected neural network, that merges
        state and action vectors together.

        Parameters
        ----------
        input_variables : list of str
            The list of input variables
        agents : list(Agent)
            The list of agents, which input is passed to the critic
        global_state_keys : list of str
            The list of all keys (in the correct order, as they occur in the input tensor later!) in the state.
        args : argparse
            Additional arguments, optional.
        """
        self.input_variables = input_variables
        self.agent_variables = [ f"{agent.name} {controlled_var}" for agent in agents for controlled_var in agent.controlled_parameters ]
        self.agent_var_sorted= [ (agent.name, agent.controlled_parameters) for agent in agents ]
        # Hint: agent_var_sorted is the dict (actually it is a list so prohibit reordering) that is used by self.forward()
        # agent_variables is only for the publicity to know that to do
        #
        self.input_state_var_size = len(input_variables)
        self.input_agent_var_size = len(self.agent_variables)
        self.input_size  = self.input_state_var_size + self.input_agent_var_size
        hidden_size      = 40 if args is None else args.critic_hidden_size
        self.hidden_size = hidden_size
        self.lr          = 0.001 if args is None else args.lr
        self.w_l2        = 0.00001 if args is None else args.critic_w_l2
        self.use_cuda    = torch.cuda.is_available() if args is None else args.use_cuda
        if not args is None and args.critic_hidden_activation == "LeakyReLU":
            activation_fx     = torch.nn.LeakyReLU
        else:
            activation_fx     = torch.nn.Tanh
        #if not args is None and args.critic_last_activation == "LeakyReLU":
        #    activation_fx_end = torch.nn.LeakyReLU
        #else:
        #    activation_fx_end = torch.nn.Tanh
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, hidden_size),
            activation_fx(),
            torch.nn.Linear(hidden_size, hidden_size),
            activation_fx(),
            torch.nn.Linear(hidden_size, 1)
        )
        self.model_target = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, hidden_size),
            activation_fx(),
            torch.nn.Linear(hidden_size, hidden_size),
            activation_fx(),
            torch.nn.Linear(hidden_size, 1)
        )
        # change initialization
        for m_param in self.model.parameters():
            if len(m_param.shape) == 1:
                # other initialization for biases
                torch.nn.init.constant_(m_param, 0.001)
            else:
                torch.nn.init.normal_(m_param, 0.0, 1.0)
        # copy weights from Q -> target Q
        for m_param, mtarget_param in zip(self.model.parameters(), self.model_target.parameters()):
            mtarget_param.data.copy_(m_param.data)
        # init optimizer and loss
        self._init_optimizer()
        self.loss      = torch.nn.MSELoss()
        # define the transformation matrix
        trafo_list = []
        for input_param in input_variables:
            idx = global_state_keys.index(input_param)
            row = torch.zeros(len(global_state_keys))
            row[idx] = 1.0
            trafo_list.append(row)
        self.trafo_matrix = torch.stack(trafo_list).T
        # cuda?
        self._init_cuda()


    def _init_optimizer(self):
        """
        Initializes the optimizers.
        """
        self.optimizer = torch.optim.Adam(params = self.model.parameters(), lr = self.lr, weight_decay = self.w_l2)


    def _init_cuda(self):
        """
        Moves all models to the GPU.
        """
        if self.use_cuda:
            self.model_target = self.model_target.to(0)
            self.model = self.model.to(0)
            self.trafo_matrix = self.trafo_matrix.to(0)
            self._init_optimizer()


    def compute_loss_and_optimize(self, q_tensor, y_tensor, no_backprop = False):
        """
        Computes the loss, backpropagates this and applies on step by the optimizer.

        y_tensor will be detached to ensure proper backpropagation to the q network only.
        """
        L = self.loss(q_tensor, y_tensor.detach())
        if not no_backprop:
            L.backward()
            self.optimizer.step()
        return L


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
        if type(state_tensor) == list:
            state_tensor = torch.cat(state_tensor, dim=0)
        if type(all_actions_tensor) == list:
            all_actions_tensor = torch.cat(all_actions_tensor, dim=1)
        # cat state tensor and action tensor
        if self.use_cuda:
            all_actions_tensor = all_actions_tensor.to(0)
            state_tensor       = state_tensor.to(0)
        state_tensor = torch.matmul(state_tensor, self.trafo_matrix).detach()
        input_ten = torch.cat([state_tensor, all_actions_tensor], dim=1)
        if no_target:
            output_tensor = self.model(input_ten)
        else:
            output_tensor = self.model_target(input_ten)
        if self.use_cuda:
            return output_tensor.cpu()
        return output_tensor


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

    def save_models_to_disk(self, storage_dir, prefix=""):
        torch.save(self.model, os.path.join(storage_dir, prefix + "_model.pickle"))
        torch.save(self.model_target, os.path.join(storage_dir, prefix + "_model_target.pickle"))

    def load_models_from_disk(self, storage_dir, prefix=""):
        self.model = torch.load(os.path.join(storage_dir, prefix + "_model.pickle"))
        self.model_target= torch.load(os.path.join(storage_dir, prefix + "_model_target.pickle"))
        self._init_optimizer()
        self._init_cuda()

