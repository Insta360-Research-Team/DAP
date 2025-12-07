import torch
import numpy as np


def get_phi_range(orig_H, crop_H, start_row):
    """
    Compute the latitude (phi) range corresponding to a cropped equirectangular region.

    Args:
        orig_H (int): Original equirectangular image height.
        crop_H (int): Cropped image height.
        start_row (int): Starting row index of the cropped region.

    Returns:
        tuple: (phi_min, phi_max) in radians.
    """
    phi_min = np.pi / 2 - np.pi * (start_row + crop_H) / orig_H
    phi_max = np.pi / 2 - np.pi * start_row / orig_H
    return phi_min, phi_max


def equirectangular_rotate_yaw(tensor, yaw_deg):
    """
    Rotate a cropped equirectangular image along the yaw (horizontal) axis.

    Args:
        tensor (torch.Tensor): Cropped equirectangular image of shape (B, C, H, W).
        yaw_deg (float): Rotation angle in degrees.

    Returns:
        torch.Tensor: Rotated equirectangular image of shape (B, C, H, W).
    """
    B, C, H, W = tensor.shape
    device = tensor.device
    dtype = tensor.dtype

    # Infer original equirectangular height and crop starting row
    orig_H = W // 2
    crop_H = H
    start_row = (orig_H - crop_H) // 2

    # Compute phi (latitude) range for the cropped region
    phi_min, phi_max = get_phi_range(orig_H, crop_H, start_row)

    # Build spherical sampling grid
    v = torch.linspace(0, H - 1, H, device=device)
    u = torch.linspace(0, W - 1, W, device=device)
    grid_v, grid_u = torch.meshgrid(v, u, indexing='ij')  # (H, W)

    # Convert pixel indices to spherical coordinates
    phi = phi_min + (phi_max - phi_min) * (grid_v / (H - 1))  # (H, W)
    theta = -np.pi + 2 * np.pi * (grid_u / (W - 1))           # (H, W)

    # Convert spherical coordinates to 3D Cartesian coordinates
    x = torch.cos(phi) * torch.sin(theta)
    y = torch.sin(phi)
    z = torch.cos(phi) * torch.cos(theta)
    xyz = torch.stack([x, y, z], dim=-1).to(device=device, dtype=dtype)  # (H, W, 3)

    # Construct rotation matrix (yaw rotation around the Y-axis)
    yaw = np.deg2rad(yaw_deg)
    rot = torch.tensor([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ]).to(device=device, dtype=dtype)  # (3, 3)

    # Apply rotation
    xyz_rot = torch.matmul(xyz, rot.T)  # (H, W, 3)

    # Convert rotated Cartesian coordinates back to spherical coordinates
    x_r, y_r, z_r = xyz_rot[..., 0], xyz_rot[..., 1], xyz_rot[..., 2]
    phi_r = torch.asin(torch.clamp(y_r, -1, 1))
    theta_r = torch.atan2(x_r, z_r)

    # Map spherical coordinates back to image coordinates
    v_r = (phi_r - phi_min) / (phi_max - phi_min) * (H - 1)
    u_r = (theta_r + np.pi) / (2 * np.pi) * (W - 1)

    # Normalize coordinates to [-1, 1] for grid_sample
    grid_u_norm = u_r / (W - 1) * 2 - 1
    grid_v_norm = v_r / (H - 1) * 2 - 1
    grid = torch.stack([grid_u_norm, grid_v_norm], dim=-1)  # (H, W, 2)
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # (B, H, W, 2)

    # Sample using bilinear interpolation
    out = torch.nn.functional.grid_sample(
        tensor, grid, mode='bilinear', padding_mode='border', align_corners=True)
    return out
