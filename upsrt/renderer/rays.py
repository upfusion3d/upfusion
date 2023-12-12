"""Utils for ray manipulation"""

import numpy as np
import torch
from pytorch3d.renderer.implicit.raysampling import RayBundle as RayBundle

from upsrt.renderer.cameras import pixel_to_ndc
from upsrt.utils.device import get_default_device
from upsrt.utils.typing import is_scalar

############################# RAY BUNDLE UTILITIES #############################

def get_grid_raybundle(cameras, image_size=(64, 64), max_depth=1,
                       min_x=1, max_x=-1, min_y=1, max_y=-1):
    if isinstance(image_size, int):
        image_width = image_size
        image_height = image_size
    else:
        image_height, image_width = image_size

    xs = torch.linspace(min_x, max_x, image_width)
    ys = torch.linspace(min_y, max_y, image_height)

    xs, ys = np.meshgrid(xs, ys)  # (image_height, image_width), (image_height, image_width)
    xs = xs.reshape(-1)  # (image_height * image_width, )
    ys = ys.reshape(-1)  # (image_height * image_width, )

    raybundle = get_directional_raybundle(cameras=cameras, x_pos_ndc=xs, y_pos_ndc=ys, max_depth=max_depth)
    return raybundle


def get_directional_raybundle(cameras, x_pos_ndc, y_pos_ndc, max_depth=1):
    if is_scalar(x_pos_ndc):
        x_pos_ndc = [x_pos_ndc]
    if is_scalar(y_pos_ndc):
        y_pos_ndc = [y_pos_ndc]
    assert is_scalar(max_depth)

    if not isinstance(x_pos_ndc, torch.Tensor):
        x_pos_ndc = torch.tensor(x_pos_ndc)  # (N, )
    if not isinstance(y_pos_ndc, torch.Tensor):
        y_pos_ndc = torch.tensor(y_pos_ndc)  # (N, )

    xy_depth = torch.stack((x_pos_ndc, y_pos_ndc, torch.ones_like(x_pos_ndc) * max_depth), dim=-1)  # (N, 3)

    num_points = xy_depth.shape[0]

    unprojected = cameras.unproject_points(xy_depth.to(cameras.device), world_coordinates=True, from_ndc=True)  # (N, 3)
    unprojected = unprojected.unsqueeze(0).to('cpu')  # (B, N, 3)

    origins = cameras.get_camera_center()[:, None, :].expand(-1, num_points, -1).to('cpu')  # (B, N, 3)
    directions = unprojected - origins  # (B, N, 3)
    directions = directions / directions.norm(dim=-1).unsqueeze(-1)  # (B, N, 3)
    lengths = torch.tensor([[0, 3]]).unsqueeze(0).expand(-1, num_points, -1).to('cpu')  # (B, N, 2)
    xys = xy_depth[:, :2].unsqueeze(0).to('cpu')  # (B, N, 2)

    raybundle = RayBundle(origins=origins.to('cpu'), directions=directions.to('cpu'),
                          lengths=lengths.to('cpu'), xys=xys.to('cpu'))
    return raybundle


def get_patch_raybundle(cameras, num_patches_x, num_patches_y, max_depth=1):
    horizontal_patch_edges = torch.linspace(1, -1, num_patches_x+1)
    # horizontal_positions = (horizontal_patch_edges[:-1] + horizontal_patch_edges[1:]) / 2  # (num_patches_x, )  # Center of patch
    horizontal_positions = horizontal_patch_edges[:-1]  # (num_patches_x,): Top left corner of patch

    vertical_patch_edges = torch.linspace(1, -1, num_patches_y+1)
    # vertical_positions = (vertical_patch_edges[:-1] + vertical_patch_edges[1:]) / 2  # (num_patches_y, )  # Center of patch
    vertical_positions = vertical_patch_edges[:-1]  # (num_patches_y,): Top left corner of patch

    h_pos, v_pos = np.meshgrid(horizontal_positions, vertical_positions)  # (num_patches_y, num_patches_x), (num_patches_y, num_patches_x)
    h_pos = h_pos.reshape(-1)  # (num_patches_y * num_patches_x)
    v_pos = v_pos.reshape(-1)  # (num_patches_y * num_patches_x)

    raybundle = get_directional_raybundle(cameras=cameras, x_pos_ndc=h_pos, y_pos_ndc=v_pos, max_depth=max_depth)
    return raybundle


def get_random_raybundle(cameras, num_rays, max_depth=1,
                         min_x=1, max_x=-1, min_y=1, max_y=-1):

    x_ndc = np.random.uniform(min_x, max_x, num_rays).astype(np.float32)
    y_ndc = np.random.uniform(min_y, max_y, num_rays).astype(np.float32)
    raybundle = get_directional_raybundle(cameras=cameras, x_pos_ndc=x_ndc, y_pos_ndc=y_ndc, max_depth=max_depth)
    return raybundle


################################ RAYS UTILITIES ################################


def process_query_ray_filter(query_ray_filter):
    """Parses an image crop mask to identify query rays only from the masked regions. Yields min and max NDC limits for query rays.

    Args:
        query_ray_filter(torch.Tensor): Tensor of shape (B, H, W) containing

    Returns:
        tuple[torch.Tensor]: Tuple of tensors of shape (B,) containing:
                                x_min_ndc: The left limit for the bounding boxes
                                x_max_ndc: The right limit for the bounding boxes
                                y_min_ndc: The top limit for the bounding boxes
                                y_max_ndc: The bottom limit for the bounding boxes
    """

    # Check if the mask is binary
    unique_values = torch.unique(query_ray_filter)
    for u in unique_values:
        assert u.data in [0, 1], 'unique values for query_ray_filter not in [0, 1]. values are = {}'.format(unique_values)

    # Get shape
    B, H, W = query_ray_filter.shape

    # Get indices of non-zero (mask) elements
    non_zero_idx = torch.nonzero(query_ray_filter)
    assert len(non_zero_idx.shape) == 2  # (N, 3)

    # Find pixel bounds of each mask
    y_min_pix, y_max_pix, x_min_pix, x_max_pix = [], [], [], []
    for b in range(B):
        # (n_b, 3) --> contains tensors like [[b, y1, x1], [b, y2, x2], ... [b, yn, xn]]
        idx_for_batch_b = non_zero_idx[non_zero_idx[:, 0]==b]
        y_min, x_min = idx_for_batch_b.amin(dim=0)[1:]
        y_max, x_max = idx_for_batch_b.amax(dim=0)[1:]
        y_min_pix.append(y_min)
        y_max_pix.append(y_max)
        x_min_pix.append(x_min)
        x_max_pix.append(x_max)

    # Convert to NDC
    x_min_ndc, y_min_ndc = pixel_to_ndc(x_min_pix, y_min_pix, H, W)
    x_max_ndc, y_max_ndc = pixel_to_ndc(x_max_pix, y_max_pix, H, W)

    return x_min_ndc, y_min_ndc, x_max_ndc, y_max_ndc


def get_patch_rays(cameras_list, num_patches_x, num_patches_y, device):
    """Returns patch rays given the camera viewpoints

    Args:
        cameras_list(list[pytorch3d.renderer.cameras.BaseCameras]): List of list of cameras (len (batch_size, num_input_views,))
        num_patches_x: Number of patches in the x-direction (horizontal)
        num_patches_y: Number of patches in the y-direction (vertical)

    Returns:
        torch.tensor: Patch rays of shape (batch_size, num_views, num_patches, 6)
    """
    batch, numviews = len(cameras_list), len(cameras_list[0])
    cameras_list = [cam for cam_batch in cameras_list for cam in cam_batch]  # Flatten the cameras list
    patch_rays = [
        get_patch_raybundle(camera, num_patches_y=num_patches_y, num_patches_x=num_patches_x)
        for camera in cameras_list
    ]

    # list of len (B * num_views), having (1, P, 6) tensors
    patch_rays = [torch.cat((patch_ray.origins, patch_ray.directions), dim=-1) for patch_ray in patch_rays]

    # patch_ray.origins -> (1, P, 3), patch_ray.directions -> (1, P, 3)
    patch_rays = torch.cat(patch_rays, dim=0)  # (B * num_views, P, 6)

    patch_rays = patch_rays.reshape(batch, numviews, num_patches_x * num_patches_y, 6).to(device)
    return patch_rays


def get_random_query_pixel_rays(
    cameras_list, num_pixel_queries, query_ray_filter, min_x,
    min_y, max_x, max_y, device, return_xys=False,
):
    """Returns query rays given the camera viewpoints

    Args:
        cameras_list(list[pytorch3d.renderer.cameras.BaseCameras]): List of len (batch_size,) containing query cameras
        num_pixel_queries(int): Number of pixel queries
        query_ray_filter(torch.Tensor|None): A tensor of shape (B, H, W) containing batch of masks within which the rays should be sampled. If None, considers the limits to be (+1, -1).

    Returns:
        torch.tensor: Query rays of shape (batch_size, num_pixel_queries, 6)
    """

    B = len(cameras_list)

    # If query ray filter (mask) is provided, then obtain X&Y limits from the mask
    if query_ray_filter is None:
        min_x_list, min_y_list, max_x_list, max_y_list = [min_x]*B, [min_y]*B, [max_x]*B, [max_y]*B
    else:
        min_x_list, min_y_list, max_x_list, max_y_list = process_query_ray_filter(query_ray_filter)

    random_query_rays = [
        get_random_raybundle(camera, num_rays=num_pixel_queries, min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y)
        for camera, min_x, min_y, max_x, max_y in zip(cameras_list, min_x_list, min_y_list, max_x_list, max_y_list)
    ]
    xys = [query_ray.xys for query_ray in random_query_rays]
    # query_ray.xys -> (1, num_queries, 2)
    xys = torch.cat(xys, dim=0).to(device)  # (batch_size, num_queries, 2)

    random_query_rays = [torch.cat((patch_ray.origins, patch_ray.directions), dim=-1) for patch_ray in random_query_rays]
    random_query_rays = torch.cat(random_query_rays, dim=0).to(device)  # (batch_size, num_queries, 6)

    assert random_query_rays.shape == (len(xys), num_pixel_queries, 6)

    if return_xys:
        return random_query_rays, xys
    else:
        return random_query_rays


def get_grid_rays(cameras_list, image_size, min_x, min_y, max_x, max_y, device):
    """Returns rays in a grid (one per pixel) parameterized as a 6-D vector (origin: x, y, z | direction: a, b, c).

    Args:
        cameras_list(list[pytorch3d.renderer.cameras.CamerasBase]): List of Pytorch3D cameras of length (n_cameras,).
        image_size(tuple[int, int]): Size of the image in pixels (height, width).
        device(torch.device): The device on which the generated rays must be cast.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Tuple containing:
                                               1) grid_rays(torch.Tensor): Tensor of shape (n_cameras, H * W, 6) denoting the encoded rays (where the
                                                                           last dimension corresponds to the 6-D parameterized representation
                                                                           (origin | direction),
                                               2) xys(torch.Tensor): Tensor of shape (n_cameras, H * W, 2) denoting the NDC coordinates of the
                                                                     point in the image through which the corresponding ray passes.
    """
    # Obtain grid raybundle: Each element in the following list corresponds to a raybundle whose attributes (origins, xys, directions, lengths) are
    # tensors of shape (1, H*W, d) where d is the dimensionality of the quantity.
    grid_rays = [
        get_grid_raybundle(camera, image_size=image_size, min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y)
        for camera in cameras_list
    ]

    # Concatenate xys (along the batch dimension) to create a single tensor
    xys = [grid_ray.xys for grid_ray in grid_rays]  # grid_rays.xys -> (1, H*W, 2)
    xys = torch.cat(xys, dim=0)  # (n_cameras, H*W, 2)

    # Concatenate origins and directions to create a single tensor
    # The final rays are represented as the 6-dimensional representation (origin|direction)
    grid_rays = [torch.cat((grid_ray.origins, grid_ray.directions), dim=-1) for grid_ray in grid_rays]
    grid_rays = torch.cat(grid_rays, dim=0).to(device)  # (n_cameras, H*W, 6)

    return grid_rays, xys


############################ RAY PARAMETERIZATION ##############################

def get_plucker_parameterization(ray):
    """Returns the plucker representation of the rays given the (origin, direction) representation

    Args:
        ray(torch.Tensor): Tensor of shape (..., 6) with the (origin, direction) representation

    Returns:
        torch.Tensor: Tensor of shape (..., 6) with the plucker (D, OxD) representation
    """
    ray = ray.clone()  # Create a clone
    ray_origins = ray[..., :3]
    ray_directions = ray[..., 3:]
    ray_directions = ray_directions / ray_directions.norm(dim=-1).unsqueeze(-1)  # Normalize ray directions to unit vectors
    plucker_normal = torch.cross(ray_origins, ray_directions, dim=-1)
    plucker_parameterization = torch.cat([ray_directions, plucker_normal], dim=-1)

    return plucker_parameterization


############################## POSITION ENCODING ###############################

def positional_encoding(ray, n_freqs=10, parameterize=None, start_freq=0):
    """
    Positional Embeddings. For more details see Section 5.1 of
    NeRFs: https://arxiv.org/pdf/2003.08934.pdf

    Args:
        ray: (B,num_input_views,P,6)
        n_freqs: num of frequency bands
        parameterize(str|None): Parameterization used for rays. Recommended: use 'plucker'. Default=None.

    Returns:
        pos_embeddings: Mapping input ray from R to R^{2*n_freqs}.
    """

    if parameterize is None:
        pass
    elif parameterize == 'plucker':
        # direction unit-normalized, (o+nd, d) has same representation as (o+md, d) [4 DOF]
        # ray_origins = ray[..., :3]
        # ray_directions = ray[..., 3:]
        # ray_directions = ray_directions / ray_directions.norm(dim=-1).unsqueeze(-1)  # Normalize ray directions to unit vectors
        # plucker_normal = torch.cross(ray_origins, ray_directions, dim=-1)
        # plucker_parameterization = torch.cat([ray_directions, plucker_normal], dim=-1)
        ray = get_plucker_parameterization(ray)
    else:
        raise NotImplementedError(f'parameterize={parameterize} not implemented.')

    freq_bands = 2. ** torch.arange(start_freq, start_freq+n_freqs) * np.pi
    sin_encodings = [torch.sin(ray * freq) for freq in freq_bands]
    cos_encodings = [torch.cos(ray * freq) for freq in freq_bands]

    pos_embeddings = torch.cat(sin_encodings + cos_encodings, dim=-1)  # B, num_input_views, P, 6 * 2n_freqs
    return pos_embeddings

##################################################################################################
# WIP
##################################################################################################

def get_grid_rays_gpu(cameras_list, image_size, min_x, min_y, max_x, max_y):
    """Returns rays in a grid (one per pixel) parameterized as a 6-D vector (origin: x, y, z | direction: a, b, c).

    Args:
        cameras_list(list[pytorch3d.renderer.cameras.CamerasBase]): List of Pytorch3D cameras of length (n_cameras,).
        image_size(tuple[int, int]): Size of the image in pixels (height, width).
        device(torch.device): The device on which the generated rays must be cast.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Tuple containing:
                                               1) grid_rays(torch.Tensor): Tensor of shape (n_cameras, H * W, 6) denoting the encoded rays (where the
                                                                           last dimension corresponds to the 6-D parameterized representation
                                                                           (origin | direction),
                                               2) xys(torch.Tensor): Tensor of shape (n_cameras, H * W, 2) denoting the NDC coordinates of the
                                                                     point in the image through which the corresponding ray passes.
    """
    # Obtain grid raybundle: Each element in the following list corresponds to a raybundle whose attributes (origins, xys, directions, lengths) are
    # tensors of shape (1, H*W, d) where d is the dimensionality of the quantity.
    grid_rays = [
        get_grid_raybundle_gpu(camera, image_size=image_size, min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y)
        for camera in cameras_list
    ]

    # Concatenate xys (along the batch dimension) to create a single tensor
    xys = [grid_ray.xys for grid_ray in grid_rays]  # grid_rays.xys -> (1, H*W, 2)
    xys = torch.cat(xys, dim=0)  # (n_cameras, H*W, 2)

    # Concatenate origins and directions to create a single tensor
    # The final rays are represented as the 6-dimensional representation (origin|direction)
    grid_rays = [torch.cat((grid_ray.origins, grid_ray.directions), dim=-1) for grid_ray in grid_rays]
    grid_rays = torch.cat(grid_rays, dim=0)  # (n_cameras, H*W, 6)

    return grid_rays, xys

def get_grid_raybundle_gpu(
        cameras, image_size=(64, 64), max_depth=1,
        min_x=1, max_x=-1, min_y=1, max_y=-1
    ):
    if isinstance(image_size, int):
        image_width = image_size
        image_height = image_size
    else:
        image_height, image_width = image_size

    xs = torch.linspace(min_x, max_x, image_width)
    ys = torch.linspace(min_y, max_y, image_height)

    xs, ys = torch.meshgrid(xs, ys, indexing="xy")  # (image_height, image_width), (image_height, image_width)
    xs = xs.reshape(-1)  # (image_height * image_width, )
    ys = ys.reshape(-1)  # (image_height * image_width, )

    raybundle = get_directional_raybundle_gpu(cameras=cameras, x_pos_ndc=xs, y_pos_ndc=ys, max_depth=max_depth)
    return raybundle

def get_directional_raybundle_gpu(cameras, x_pos_ndc, y_pos_ndc, max_depth=1):

    assert is_scalar(max_depth)

    if not isinstance(x_pos_ndc, torch.Tensor):
        raise ValueError
    if not isinstance(y_pos_ndc, torch.Tensor):
        raise ValueError

    xy_depth = torch.stack((x_pos_ndc, y_pos_ndc, torch.ones_like(x_pos_ndc) * max_depth), dim=-1)  # (N, 3)

    num_points = xy_depth.shape[0]

    unprojected = cameras.unproject_points(xy_depth.to(cameras.device), world_coordinates=True, from_ndc=True)  # (N, 3)
    unprojected = unprojected.unsqueeze(0)  # (B, N, 3)

    origins = cameras.get_camera_center()[:, None, :].expand(-1, num_points, -1)  # (B, N, 3)
    directions = unprojected - origins  # (B, N, 3)
    directions = directions / directions.norm(dim=-1).unsqueeze(-1)  # (B, N, 3)
    lengths = torch.tensor([[0, 3]]).unsqueeze(0).expand(-1, num_points, -1)  # (B, N, 2)
    xys = xy_depth[:, :2].unsqueeze(0)  # (B, N, 2)

    raybundle = RayBundle(origins=origins, directions=directions,
                          lengths=lengths, xys=xys)
    return raybundle