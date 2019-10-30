import torch
import torch.nn as nn


def tensor_to_device(x, device):
    if device is None:
        return x
    return _tensor_to_device(x, device)


def _tensor_to_device(x, device):
    """
    Convert without checking 'device'
    """
    if x is None:
        return None
    if torch.is_tensor(x) or isinstance(x, nn.Module):
        return x.to(device, non_blocking=True)
    return [tensor_to_device(component, device) for component in x]
