import torch


def transform_rays(reference_R, reference_T, rays):
    '''
    PyTorch3D Convention is used: X_cam = X_world @ R + T

    Args:
        reference_R: world2cam rotation matrix for reference camera (B, 3, 3)
        reference_T: world2cam translation vector for reference camera (B, 3)
        rays: (origin, direction) defined in world reference frame (B, V, N, 6)
    Returns:
        torch.Tensor: Transformed rays w.r.t. reference camera (B, V, N, 6)
    '''
    batch, num_views, num_rays, ray_dim = rays.shape
    assert ray_dim == 6, \
        'First 3 dimensions should be origin; Last 3 dimensions should be direction'

    rays = rays.reshape(batch, num_views*num_rays, ray_dim)
    rays_out = rays.clone()
    rays_out[..., :3] = torch.bmm(rays[..., :3], reference_R) + reference_T.unsqueeze(-2)
    rays_out[..., 3:] = torch.bmm(rays[..., 3:], reference_R)
    rays_out = rays_out.reshape(batch, num_views, num_rays, ray_dim)
    return rays_out


def plucker_dist(ray1, ray2, eps=1e-6):
    # Plucker ray is represented as (l, m),
    # l is direction unit norm, m = (oxl)

    # ray1 (l1, m1): (B, Q, 6)
    # ray2 (l2, m2): (B, P, 6)

    Q = ray1.shape[1]
    P = ray2.shape[1]

    ray1 = ray1.unsqueeze(2).repeat(1, 1, P, 1)  # (B, Q, P, 6)
    ray2 = ray2.unsqueeze(1).repeat(1, Q, 1, 1)  # (B, Q, P, 6)

    # (l1, m1) * (l2, m2) = l1.m2 + l2.m1
    reci_prod = ((ray1[..., :3] * ray2[..., 3:]).sum(-1) + \
                (ray1[..., 3:] * ray2[..., :3]).sum(-1)).abs()  # (B, Q, P)

    # || l1 x l2 ||
    l1_cross_l2 = torch.cross(ray1[..., :3], ray2[..., :3], dim=-1)  # (B, Q, P, 3)
    l1_cross_l2_norm = l1_cross_l2.norm(dim=-1) # (B, Q, P)

    # || l1 x (m1-m2)/s ||
    # s = ray2[..., :3] / ray1[..., :3]  # (B, Q, P, 3)
    # s = s.mean(dim=-1).unsqueeze(-1)  # (B, Q, P, 1)
    s = 1
    l1_cross_m1_minus_m2 = torch.cross(ray1[..., :3], (ray1[..., 3:] - ray2[..., 3:])/s)
    l1_cross_m1_minus_m2_norm = l1_cross_m1_minus_m2.norm(dim=-1) # (B, Q, P)

    # ||l1||^2
    l1_norm_sq = torch.norm(ray1[..., :3], dim=-1) ** 2 # (B, Q, P)

    distance = l1_cross_m1_minus_m2_norm / (l1_norm_sq + eps) # (B, Q, P)
    mask = (l1_cross_l2_norm > eps)
    distance[mask] = reci_prod[mask] / (l1_cross_l2_norm[mask] + eps)

    return distance
