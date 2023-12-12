import cv2
import imageio
import numpy as np
import torch

def convert_tensor_to_images(input_views, query_img, pred_img, unnorm_query = False, unnorm_pred = False):
    """
    input_views: (V, 3, H, W), pixel range: [-1, 1]
    query_img: (1, 3, H, W), pixel range: [0, 1]
    pred_img: (1, H, W, 3), pixel range: [0, 1]
    """
    input_image = np.transpose(input_views.cpu().numpy(), (0, 2, 3, 1)) * 0.5 + 0.5 # (V, H, W, 3) - [0, 1]

    if unnorm_query:
        query_image = np.transpose(query_img.cpu().numpy(), (0, 2, 3, 1)) * 0.5 + 0.5 # (1, H, W, 3) - [0, 1]
    else:
        query_image = np.transpose(query_img.cpu().numpy(), (0, 2, 3, 1)) # (1, H, W, 3) - [0, 1]

    if unnorm_pred:
        pred_image = pred_img.cpu().numpy() * 0.5 + 0.5 # (1, H, W, 3) - [0, 1]
    else:
        pred_image = pred_img.cpu().numpy() # (1, H, W, 3) - [0, 1]

    return input_image, query_image, pred_image

def stitch_images(input_image, query_image, pred_image):
    """
    input_image: (V, H, W, 3) - [0, 1]
    query_image: (1, H, W, 3) - [0, 1]
    pred_image: (1, H, W, 3) - [0, 1]
    """
    num_input_images = input_image.shape[0]
    _, hq, wq, _ = query_image.shape

    # Assuming images are numpy arrays of shape N x H x W x 3
    input_image = np.concatenate(list(input_image), axis=1) # H x Ni*W x 3
    if input_image.shape[1:3] != query_image.shape[1:3]:
        input_image = cv2.resize(input_image, dsize=(wq * num_input_images, hq))
    query_image = np.concatenate(list(query_image), axis=1) # H x Nq*W x3
    pred_image = np.concatenate(list(pred_image), axis=1) # H x Np*W x 3

    stitched_image = np.concatenate([input_image, query_image, pred_image], axis=1)
    return stitched_image

def save_images_as_gif(save_path, images, fps=5):
    imageio.mimsave(save_path, images, fps=fps)

# ===================== SAMPLING UTILITIES =====================

def sample_images_at_xy(images, xy_grid):
    """ Samples the provided images at the given locations along a grid
    Args:
        images(torch.Tensor): Batch of images of shape (B, 3, H, W).
        xy_grid(torch.Tensor): Grid of shape (B, h, w, 2), or, (B, n, 2) containing the NDC coordinates to sample the image from.
    Returns:
        torch.Tensor: The sampled image of shape (B, 3, h, w), or (B, 3, n)
    """
    assert images.shape[0] == xy_grid.shape[0], \
        'batch_size for images and xy_grid does not match.'
    batch_size = images.shape[0]

    if len(xy_grid.shape) == 3:  # (B, n, 2)
        xy_grid = -xy_grid.view(batch_size, -1, 1, 2)  # (B, n, 1, 2)

        images_sampled = torch.nn.functional.grid_sample(
            images,   # (B, 3, H, W)
            xy_grid,  # (B, n, 1, 2)
            align_corners=True,
            mode="bilinear",
        )  # (B, 3, n, 1)

        images_sampled = images_sampled.squeeze(3)  # (B, 3, n)

    elif len(xy_grid.shape) == 4:
        xy_grid = -xy_grid  # (B, h, w, 2)

        images_sampled = torch.nn.functional.grid_sample(
            images,  # (B, 3, H, W)
            xy_grid,  # (B, h, w, 2)
            align_corners=True,
            mode="bilinear",
        )  # (B, 3, h, w)

    else:
        raise ValueError(f'Expected xy_grid of shape (B, n, 2) or (B, h, w, 2), but got the shape {xy_grid.shape}.')

    return images_sampled
