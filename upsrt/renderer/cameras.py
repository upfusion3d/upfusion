import numpy as np


def pixel_to_ndc(x, y, height, width):
    """ Converts Pixel (screen-space) coordinates to NDC coordinates

    The convention used is as follows:
        * Screen Space - X: [0, W-1], Y: [0, H-1]
        * NDC Space - X: [1, -1], Y: [1, -1]

    where (0, 0) and (H, W) are the top-left and bottom right corners of the image in the screen space (pixel coordinates), and, (1, 1) and (-1,
    -1) are the top-left and the bottom-right corners in the NDC space (NDC coordinates) respectively.

    Note that the output of this function is a numpy array (loses differentiability).

    Args:
        x(int|float|list|np.array): A scalar or a list (or array) of values indicating the x-coordinate(s) in the screen space
        y(int|float|list|np.array): A scalar or a list (or array) of values indicating the y-coordinate(s) in the screen space
        height(int): The height of the image in screen space (pixels)
        width(int): The width of the image in screen space (pixels)

    Returns:
        tuple[np.arraym, np.array]: A tuple containing:
                                        1) x_ndc(np.ndarray): The NDC coordinates corresponding to the provided x-coordinates (pixel coords.),
                                        2) y_ndc(np.ndarray): The NDC coordinates corresponding to the provided y-coordinates (pixel coords.)

    """
    if isinstance(x, float) or isinstance(x, int):
        x = np.array([x], dtype=np.int32)
    elif isinstance(x, list):
        x = np.array(x, dtype=np.int32)

    if isinstance(y, float) or isinstance(y, int):
        y = np.array([y], dtype=np.int32)
    elif isinstance(y, list):
        y = np.array(y, dtype=np.int32)

    x_ndc = np.linspace(1, -1, width, dtype=np.float32)[x]
    y_ndc = np.linspace(1, -1, height, dtype=np.float32)[y]
    return x_ndc, y_ndc
