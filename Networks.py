
#
# Function for generating torch-networks
#

import torch
import collections

def generate_network(network_type, input_size, output_size, use_layer_norm = False, add_tanh = False):

    if network_type == "2HiddenLayer,Trapezium":
        hidden_size1 = max(19, input_size)
        hidden_size2 = (hidden_size1 + 2*output_size) // 3
        model = collections.OrderedDict()
        model["linear-1"]    = torch.nn.Linear(input_size,   hidden_size1)
        if use_layer_norm:
            model["lnorm-1"] = torch.nn.LayerNorm(hidden_size1)
        model["leakyReLU-1"] = torch.nn.LeakyReLU()
        model["linear-2"]    = torch.nn.Linear(hidden_size1, hidden_size2)
        if use_layer_norm:
            model["lnorm-2"] = torch.nn.LayerNorm(hidden_size2)
        model["leakyReLU-2"] = torch.nn.LeakyReLU()
        model["linear-3"]    = torch.nn.Linear(hidden_size2, output_size)
        if add_tanh:
            model["tanh-3"]  = torch.nn.Tanh()
        return torch.nn.Sequential(model)

    if network_type == "2HiddenLayer,Trapezium,SiLU":
        hidden_size1 = max(19, input_size)
        hidden_size2 = (hidden_size1 + 2*output_size) // 3
        model = collections.OrderedDict()
        model["linear-1"]    = torch.nn.Linear(input_size,   hidden_size1)
        if use_layer_norm:
            model["lnorm-1"] = torch.nn.LayerNorm(hidden_size1)
        model["SiLU-1"]      = torch.nn.SiLU()
        model["linear-2"]    = torch.nn.Linear(hidden_size1, hidden_size2)
        if use_layer_norm:
            model["lnorm-2"] = torch.nn.LayerNorm(hidden_size2)
        model["SiLU-2"]      = torch.nn.SiLU()
        model["linear-3"]    = torch.nn.Linear(hidden_size2, output_size)
        if add_tanh:
            model["tanh-3"]  = torch.nn.Tanh()
        return torch.nn.Sequential(model)

    if network_type == "1HiddenBigLayer":
        hidden_size1 = 3 * max(input_size, output_size)
        model = collections.OrderedDict()
        model["linear-1"]    = torch.nn.Linear(input_size,   hidden_size1)
        if use_layer_norm:
            model["lnorm-1"] = torch.nn.LayerNorm(hidden_size1)
        model["leakyReLU-1"] = torch.nn.LeakyReLU()
        model["linear-2"]    = torch.nn.Linear(hidden_size1, output_size)
        if add_tanh:
            model["tanh-2"]  = torch.nn.Tanh()
        return torch.nn.Sequential(model)

    if network_type == "1HiddenBigLayer,ELU":
        hidden_size1 = 3 * max(input_size, output_size)
        model = collections.OrderedDict()
        model["linear-1"]    = torch.nn.Linear(input_size,   hidden_size1)
        if use_layer_norm:
            model["lnorm-1"] = torch.nn.LayerNorm(hidden_size1)
        model["ELU-1"]       = torch.nn.ELU(alpha=0.9)
        model["linear-2"]    = torch.nn.Linear(hidden_size1, output_size)
        if add_tanh:
            model["tanh-2"]  = torch.nn.Tanh()
        return torch.nn.Sequential(model)

    if network_type == "1HiddenBigLayer,SiLU":
        hidden_size1 = 3 * max(input_size, output_size)
        model = collections.OrderedDict()
        model["linear-1"]    = torch.nn.Linear(input_size,   hidden_size1)
        if use_layer_norm:
            model["lnorm-1"] = torch.nn.LayerNorm(hidden_size1)
        model["SiLU-1"]      = torch.nn.SiLU()
        model["linear-2"]    = torch.nn.Linear(hidden_size1, output_size)
        if add_tanh:
            model["tanh-2"]  = torch.nn.Tanh()
        return torch.nn.Sequential(model)

    else:
        raise AttributeError(f"Unknown network type '{network_type}'.")

