"""Handling device configuration and management"""

import torch

DEFAULT_GPU_DEVICE_ID = 0

def get_default_device():
    """Returns the default torch.device object (GPU if available, else CPU)
    """
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{DEFAULT_GPU_DEVICE_ID}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    return device