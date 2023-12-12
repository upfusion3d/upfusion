import numpy as np
import torch


def is_scalar(x):
    """Returns True if the provided variable is a scalar

    Args:
        x: scalar or array-like (numpy array or torch tensor)

    Returns:
        bool: True if x is of the type scalar, or array-like with 0 dimension. False, otherwise

    """
    if isinstance(x, float) or isinstance(x, int):
        return True

    if isinstance(x, np.ndarray) and np.ndim(x) == 0:
        return True

    if isinstance(x, torch.Tensor) and x.dim() == 0:
        return True

    return False