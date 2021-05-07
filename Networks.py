
#
# Function for generating torch-networks
#

import torch

def generate_network(network_type, input_size, output_size):

    if network_type == "2HiddenLayer,Trapezium":
        hidden_size1 = max(19, input_size)
        hidden_size2 = (hidden_size1 + 2*output_size) // 3
        return torch.nn.Sequential(
                torch.nn.Linear(input_size,   hidden_size1),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(hidden_size1, hidden_size2),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(hidden_size2, output_size)
            )

    if network_type == "2HiddenLayer,Trapezium,SiLU":
        hidden_size1 = max(19, input_size)
        hidden_size2 = (hidden_size1 + 2*output_size) // 3
        return torch.nn.Sequential(
                torch.nn.Linear(input_size,   hidden_size1),
                torch.nn.SiLU(),
                torch.nn.Linear(hidden_size1, hidden_size2),
                torch.nn.SiLU(),
                torch.nn.Linear(hidden_size2, output_size)
            )

    if network_type == "1HiddenBigLayer":
        hidden_size1 = 3 * max(input_size, output_size)
        return torch.nn.Sequential(
                torch.nn.Linear(input_size,   hidden_size1),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(hidden_size1, output_size)
            )

    if network_type == "1HiddenBigLayer,ELU":
        hidden_size1 = 3 * max(input_size, output_size)
        return torch.nn.Sequential(
                torch.nn.Linear(input_size,   hidden_size1),
                torch.nn.ELU(alpha=0.9),
                torch.nn.Linear(hidden_size1, output_size)
            )

    if network_type == "1HiddenBigLayer,SiLU":
        hidden_size1 = 3 * max(input_size, output_size)
        return torch.nn.Sequential(
                torch.nn.Linear(input_size,   hidden_size1),
                torch.nn.SiLU(),
                torch.nn.Linear(hidden_size1, output_size)
            )

    else:
        raise AttributeError(f"Unknown network type '{network_type}'.")

