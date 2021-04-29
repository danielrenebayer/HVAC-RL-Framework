
import torch

def get_optimizer_from_args(parameters, optimizer, lr, weight_decay):
    """
    This function returns a new optimizer object from an set of arguments.
    """
    if optimizer == "sgd":
        return torch.optim.SGD(parameters,
                lr = lr,
                weight_decay = weight_decay)
    elif optimizer == "adam":
        return torch.optim.Adam(parameters,
                lr = lr,
                weight_decay = weight_decay)
    else:
        return torch.optim.RMSprop(parameters,
                lr = lr,
                weight_decay = weight_decay)


