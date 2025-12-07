import torch
import numpy as np


def get_phi_range(orig_H, crop_H, start_row):
    """
    Compute the latitude (phi) range for a cropped equirectangular region.

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


def get_cube_map_grid(face_size, face, device=None, dtype=None):
    """
    Generate a 3D direction grid for a specific cubemap face.

    Args:
        face_size (int): Resolution (height/width) of each cube face.
        face (int): Face index, in {0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z}.
        device (torch.device, optional): Device for the output tensor.
        dtype (torch.dtype, optional): Data type for the output tensor.

    Returns:
        torch.Tensor: Normalized direction vectors of shape (face_size, face_size, 3).
    """
    # Define sampling range in [-1, 1]
    rng = torch.linspace(-1, 1, face_size, device=device, dtype=dtype)
    xx, yy = torch.meshgrid(rng, -rng, indexing='ij')
    ones = torch.ones_like(xx)

    # Define direction vectors for each cube face
    if face == 0:      # +X
        xyz = torch.stack([ones, yy, xx], dim=-1)
    elif face == 1:    # -X
        xyz = torch.stack([-ones, yy, -xx], dim=-1)
    elif face == 2:    # +Y
        xyz = torch.stack([xx, ones, yy], dim=-1)
    elif face == 3:    # -Y
        xyz = torch.stack([xx, -ones, -yy], dim=-1)
    elif face == 4:    # +Z
        xyz = torch.stack([xx, yy, ones], dim=-1)
    elif face == 5:    # -Z
        xyz = torch.stack([-xx, yy, -ones], dim=-1)

    # Normalize direction vectors
    return xyz / torch.norm(xyz, dim=-1, keepdim=True)


def cube_map_from_equirectangular(equi):
    """
    Convert a cropped equirectangular image to cubemap faces.

    Args:
        equi (torch.Tensor): Cropped equirectangular image tensor of shape (B, C, crop_H, W).

    Returns:
        torch.Tensor: Cubemap tensor of shape (B, 6, C, face_size, face_size).
    """
    B, C, crop_H, W = equi.shape
    face_size = crop_H // 2

    # Infer the original equirectangular height and crop starting row
    orig_H = W // 2
    start_row = (orig_H - crop_H) // 2
    phi_min, phi_max = get_phi_range(orig_H, crop_H, start_row)

    device = equi.device
    dtype = equi.dtype
    faces = []

    for face in range(6):
        # Generate 3D direction grid for this face
        xyz = get_cube_map_grid(face_size, face, device=device, dtype=dtype)  # (face_size, face_size, 3)
        x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]

        # Convert 3D coordinates to spherical angles
        theta = torch.atan2(x, z)  # [-pi, pi]
        phi = torch.asin(y)        # [-pi/2, pi/2]

        # Map spherical coordinates to equirectangular coordinates
        v = (phi - phi_min) / (phi_max - phi_min) * (crop_H - 1)
        u = (theta + np.pi) / (2 * np.pi) * (W - 1)

        # Normalize coordinates to [-1, 1] for grid_sample
        grid_u = u / (W - 1) * 2 - 1
        grid_v = v / (crop_H - 1) * 2 - 1
        grid = torch.stack([grid_u, grid_v], dim=-1)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # (B, face_size, face_size, 2)

        # Sample from the equirectangular image
        face_img = torch.nn.functional.grid_sample(
            equi, grid, mode='bilinear', padding_mode='border', align_corners=True
        )  # (B, C, face_size, face_size)

        # Rotate 90° counterclockwise
        face_img = torch.rot90(face_img, k=1, dims=(-2, -1))
        faces.append(face_img)

    # Reorder and stack all faces
    indices = [0, 1, 2, 3, 4, 5]
    faces = [faces[i] for i in indices]
    faces = torch.stack(faces, dim=1)  # (B, 6, C, face_size, face_size)
    return faces
