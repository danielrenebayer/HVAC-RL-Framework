
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


def init_tensor(tensor, args):
    if args is None:
        torch.nn.init.xavier_normal_(m_param, 0.8)
        return
    if args.agent_init_fn == "xavier_normal":
        torch.nn.init.xavier_normal_(tensor, gain=args.agent_init_gain)
    elif args.agent_init_fn == "he_normal":
        torch.nn.init.kaiming_normal_(tensor)
    elif args.agent_init_fn == "normal":
        torch.nn.init.normal_(tensor, mean=args.agent_init_mean, std=agent_init_std)


