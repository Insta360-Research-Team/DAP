import numpy as np
import cv2
import os
from plyfile import PlyData, PlyElement
import utils3d   # 你本地已有的库
import torch

def spherical_uv_to_directions(uv: np.ndarray):
    theta, phi = (1 - uv[..., 0]) * (2 * np.pi), uv[..., 1] * np.pi
    directions = np.stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ], axis=-1)
    return directions

def spherical_uv_to_directions_torch(uv: torch.Tensor):
    """
    torch版本的球面UV坐标转方向向量
    Args:
        uv: UV坐标，形状为 [H, W, 2] 或 [B, H, W, 2]
    Returns:
        directions: 方向向量，形状为 [H, W, 3] 或 [B, H, W, 3]
    """
    theta = (1 - uv[..., 0]) * (2 * torch.pi)
    phi = uv[..., 1] * torch.pi
    directions = torch.stack([
        torch.sin(phi) * torch.cos(theta),
        torch.sin(phi) * torch.sin(theta),
        torch.cos(phi)
    ], dim=-1)
    return directions

def save_3d_points(points: np.array, colors: np.array, mask: np.array, filename: str):
    points = points.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    mask = mask.reshape(-1)

    vertex_data = np.empty(mask.sum(), dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    ])
    vertex_data['x'] = points[mask, 0]
    vertex_data['y'] = points[mask, 1]
    vertex_data['z'] = points[mask, 2]
    vertex_data['red'] = colors[mask, 0]
    vertex_data['green'] = colors[mask, 1]
    vertex_data['blue'] = colors[mask, 2]

    vertex_element = PlyElement.describe(vertex_data, 'vertex', comments=['point cloud'])
    PlyData([vertex_element], text=True).write(filename)

def depth2pointcloud(depth_path: str, image_path: str, out_ply: str):
    # 读取深度图
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"无法读取深度图: {depth_path}")

# 如果读出来是三通道，把它转成灰度
    if depth.ndim == 3:
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)

    depth = depth.astype(np.float32)
    # 将深度归一化到合理范围（假设输入是16位或8位）
    if depth.dtype == np.uint8:
        depth = depth / 255.0
    elif depth.dtype == np.uint16:
        depth = depth / 65535.0

    h, w = depth.shape
    uv = utils3d.numpy.image_uv(width=w, height=h)   # [H,W,2]
    dirs = spherical_uv_to_directions(uv)           # [H,W,3]
    points = depth[..., None] * dirs                # [H,W,3]

    # 读取颜色图
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"无法读取原图: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.shape[:2] != (h, w):
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)

    mask = depth > 0
    save_3d_points(points, image, mask, out_ply)
    print(f"✅ 点云已保存到 {out_ply}")


def depth2pts(depth):
    # 读取深度图
    depth.squeeze(1) # B,1,H,W -> B,H,W (范围是0-1)

    depth = depth.astype(np.float32)

    b, h, w = depth.shape
    
    # 生成UV坐标 [H,W,2]
    uv = utils3d.numpy.image_uv(width=w, height=h)
    # 转换为球面方向 [H,W,3]
    dirs = spherical_uv_to_directions(uv)
    
    # 广播计算3D点云 [B,H,W,3]
    points = depth[..., None] * dirs[None, ...]  # [B,H,W,1] * [1,H,W,3] = [B,H,W,3]
    
    return points

def depth2pts_torch(depth):
    """
    将深度图转换为3D点云，支持batch处理（torch版本）
    Args:
        depth: 深度图torch tensor，形状为 [B, H, W] 或 [B, 1, H, W]，值范围0-1
    Returns:
        points: 3D点云torch tensor，形状为 [B, H, W, 3]
    """
    # 确保输入是float32类型
    depth = depth.float()
    
    # 如果输入是 [B, 1, H, W]，则压缩为 [B, H, W]
    if depth.dim() == 4 and depth.shape[1] == 1:
        depth = depth.squeeze(1)  # B,1,H,W -> B,H,W
    
    b, h, w = depth.shape
    
    # 生成UV坐标 [H,W,2]
    uv = utils3d.numpy.image_uv(width=w, height=h)
    # 转换为torch tensor并移动到相同设备
    uv = torch.from_numpy(uv).to(depth.device).to(depth.dtype)
    
    # 转换为球面方向 [H,W,3]
    dirs = spherical_uv_to_directions_torch(uv)
    
    # 广播计算3D点云 [B,H,W,3]
    points = depth[..., None] * dirs[None, ...]  # [B,H,W,1] * [1,H,W,3] = [B,H,W,3]
    
    return points



if __name__ == "__main__":
    # depth_path = "/home/tione/notebook/home/wenxuan/PanDA_dualhead_alltrain/visual_nonormalloss/depth/depth_save_2.png"   # 你的深度 PNG
    # image_path = "/home/tione/notebook/home/wenxuan/PanDA_dualhead_alltrain/visual_nonormalloss/rgb/rgb_save_2.png"        # 你的原图 PNG
    # out_ply = "/home/tione/notebook/home/wenxuan/PanDA_dualhead_alltrain/visual_nonormalloss/pts/scene001_points.ply"
    path_ = "/home/tione/notebook/home/wenxuan/DAM_dualhead_alltrain/vis_exp2_insta820/rgb/"
    for file in os.listdir(path_):
        image_path = os.path.join(path_, file)
        depth_path = os.path.join(path_.replace("rgb", "depth"), file.replace("rgb", "depth"))
        os.makedirs(path_.replace("rgb", "pts"), exist_ok=True)
        out_ply = os.path.join(path_.replace("rgb", "pts"), file.replace(".png", ".ply"))
        depth2pointcloud(depth_path, image_path, out_ply)
        
    # os.makedirs("output", exist_ok=True)
    # depth2pointcloud(depth_path, image_path, out_ply)
