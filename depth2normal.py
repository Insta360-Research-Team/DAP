import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import utils3d  # 你原来的工具库
import os
# ----------------------------
# 工具函数
# ----------------------------
def spherical_uv_to_directions(uv: np.ndarray):
    theta, phi = (1 - uv[..., 0]) * (2 * np.pi), uv[..., 1] * np.pi
    directions = np.stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ], axis=-1)
    return directions


def spherical_uv_to_directions_torch(uv: torch.Tensor, device: str = 'cuda'):
    theta, phi = (1 - uv[..., 0]) * (2 * np.pi), uv[..., 1] * np.pi
    directions = torch.stack([
        torch.sin(phi) * torch.cos(theta),
        torch.sin(phi) * torch.sin(theta),
        torch.cos(phi)
    ], axis=-1).to(device)
    return directions


def normal_normalize(normal: np.ndarray):
    normal_norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    normal_norm[normal_norm < 1e-6] = 1e-6
    return normal / normal_norm


def normal_normalize_torch(normal: torch.Tensor):
    normal_norm = torch.norm(normal, dim=-1, keepdim=True)
    normal_norm = torch.where(
        normal_norm < 1e-6,
        torch.tensor(1e-6, device=normal_norm.device, dtype=normal_norm.dtype),
        normal_norm
    )
    return normal / normal_norm


def normal_to_rgb(normal: np.ndarray | torch.Tensor, normal_mask: np.ndarray | torch.Tensor = None):
    """ normal ([-1,1]) → RGB ([0,255]) """
    if torch.is_tensor(normal):
        normal = normal.detach().cpu().numpy()
        if normal_mask is not None:
            normal_mask = normal_mask.detach().cpu().numpy()

    normal_rgb = (((normal + 1) * 0.5) * 255).astype(np.uint8)

    if normal_mask is not None:
        normal_mask_c = np.stack([normal_mask]*3, axis=-1).astype(np.uint8)
        normal_rgb = normal_rgb * normal_mask_c

    return normal_rgb


# ----------------------------
# 深度转法线 (numpy版)
# ----------------------------
def depth2normal(depth: np.ndarray, mask: np.ndarray = None, to_rgb: bool = False):
    h, w = depth.shape[:2]
    # depth → 三维点
    points = depth[:, :, None] * spherical_uv_to_directions(utils3d.numpy.image_uv(width=w, height=h))

    if mask is None:
        mask = np.ones_like(depth, dtype=bool)

    normal, normal_mask = utils3d.numpy.points_to_normals(points, mask)

    # 调整方向 & normalize
    normal = normal * np.array([-1, -1, 1])
    normal = normal_normalize(normal)

    # 重排通道 (和你原代码一致)
    normal = np.stack([normal[..., 0], normal[..., 2], normal[..., 1]], axis=-1)

    if to_rgb:
        return normal, normal_mask, Image.fromarray(normal_to_rgb(normal, normal_mask))
    else:
        return normal, normal_mask


# ----------------------------
# 深度转法线 (torch版)
# ----------------------------
def depth2normal_torch(depth: torch.Tensor, mask: torch.Tensor = None, to_rgb: bool = False):
    h, w = depth.shape[-2:]
    points = depth.unsqueeze(-1) * spherical_uv_to_directions_torch(utils3d.torch.image_uv(width=w, height=h), device=depth.device)

    if mask is None:
        mask = torch.ones_like(depth, dtype=torch.uint8)

    normal, normal_mask = utils3d.torch.points_to_normals(points, mask)

    # 调整方向
    normal = normal * torch.tensor([-1, -1, 1], device=normal.device, dtype=normal.dtype)
    normal = normal_normalize_torch(normal)

    # 重排通道
    normal = torch.stack([normal[..., 0], normal[..., 2], normal[..., 1]], dim=-1)

    if to_rgb:
        normal_mask_img = normal_mask.squeeze()
        normal_imgs = [Image.fromarray(normal_to_rgb(normal[i], normal_mask_img[i])) for i in range(normal.shape[0])]
        return normal, normal_mask, normal_imgs
    else:
        return normal, normal_mask


# ----------------------------
# 主程序
# ----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--img-path', default='', type=str)

    args = parser.parse_args()


    #args.img_path是一个文件夹，文件夹中包含多个深度图
    save_out = os.path.dirname(args.img_path) + '/normal'
    os.makedirs(save_out, exist_ok=True)
    for depth_path in os.listdir(args.img_path):
        depth_path = os.path.join(args.img_path, depth_path)
        
        depth = np.load(depth_path).astype(np.float32)
        # depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255

        normal, mask, normal_img = depth2normal(depth, to_rgb=True)
        normal_img.save(os.path.join(save_out, depth_path.split('/')[-1].replace('.npy', '.png')))
        # normal_img.save(os.path.join(save_out, depth_path.split('/')[-1]))