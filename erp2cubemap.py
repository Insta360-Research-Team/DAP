#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ERP (equirectangular panorama, 2:1) -> Cube Map (6 faces)

依赖:
  pip install opencv-python numpy

用法示例:
  python erp2cubemap.py input.jpg --size 1024 --out out_dir
  python erp2cubemap.py input.jpg --size 1024 --layout cross --out cube_cross.png
"""
import os
import math
import argparse
import numpy as np
import cv2


def build_face_map(face_size, face):
    """
    为指定面构建从 cube-face 像素到 ERP 的映射。
    返回: map_x, map_y (float32, 用于 cv2.remap)
    约定的面及朝向（右手坐标, X 右, Y 上, Z 向前）:
      +X: right
      -X: left
      +Y: top
      -Y: bottom
      +Z: front
      -Z: back
    每个面 FOV = 90°，u,v ∈ [-1,1] 覆盖该面。
    """
    # 像素网格中心采样
    s = face_size
    jj, ii = np.meshgrid(np.arange(s, dtype=np.float32),
                         np.arange(s, dtype=np.float32))
    # 将像素坐标映射到 [-1,1]（中心为0），保证像素中心采样
    a = 2.0 * (jj + 0.5) / s - 1.0
    b = 2.0 * (ii + 0.5) / s - 1.0

    # 为不同面定义方向向量 (dx, dy, dz)
    if face == 'right':      # +X
        dx, dy, dz = np.ones_like(a), -b, -a
    elif face == 'left':     # -X
        dx, dy, dz = -np.ones_like(a), -b, a
    elif face == 'top':      # +Y
        dx, dy, dz = a, np.ones_like(a), b
    elif face == 'bottom':   # -Y
        dx, dy, dz = a, -np.ones_like(a), -b
    elif face == 'front':    # +Z
        dx, dy, dz = a, -b, np.ones_like(a)
    elif face == 'back':     # -Z
        dx, dy, dz = -a, -b, -np.ones_like(a)
    else:
        raise ValueError(f'Unknown face: {face}')

    # 归一化方向
    norm = np.sqrt(dx*dx + dy*dy + dz*dz)
    dx /= norm; dy /= norm; dz /= norm

    # ERP 映射（经纬 -> 像素）
    # 经度 theta ∈ (-pi, pi]，按 X->Z 的 atan2；纬度 phi ∈ [-pi/2, pi/2]
    theta = np.arctan2(dz, dx)          # 水平角：朝 +Z 为 0 -> 正确前向（front）
    phi   = np.arcsin(dy)               # 垂直角：+Y 为 +pi/2 顶部

    # 输出 map_x, map_y 是 ERP 图像坐标（列x/行y）
    # 假设输入宽 W, 高 H：
    #   x = (theta + pi) / (2*pi) * W
    #   y = (pi/2 - phi) / pi * H   (phi=+pi/2 -> y=0 顶部)
    # W,H 先占位，后面 remap 时会依据实际尺寸使用比例缩放
    # 为了支持任意输入尺寸，我们先输出归一化坐标 [0,1)，再在 remap 前乘以 W/H
    map_x_norm = (theta + math.pi) / (2.0 * math.pi)
    map_y_norm = (math.pi/2 - phi) / math.pi

    return map_x_norm.astype(np.float32), map_y_norm.astype(np.float32)


def remap_face(erp_img, map_x_norm, map_y_norm, interp, border):
    H, W = erp_img.shape[:2]
    map_x = map_x_norm * (W - 1)
    map_y = map_y_norm * (H - 1)
    return cv2.remap(erp_img, map_x, map_y, interpolation=interp, borderMode=border)


def save_six_faces(faces_dict, out_dir, base):
    os.makedirs(out_dir, exist_ok=True)
    for name, img in faces_dict.items():
        cv2.imwrite(os.path.join(out_dir, f"{base}_{name}.png"), img)


def make_cross_layout(faces, face_size):
    """
    生成常见 4x3 横向十字拼图：
         [    ][top ][    ][    ]
         [left][front][right][back]
         [    ][bottom][    ][    ]
    画布大小: (3H, 4W) = (3S, 4S)，空白填充黑色。
    """
    S = face_size
    canvas = np.zeros((3*S, 4*S, 3), dtype=np.uint8)

    # 放置
    def put(name, row, col):
        canvas[row*S:(row+1)*S, col*S:(col+1)*S] = faces[name]

    put('top',    0, 1)
    put('left',   1, 0)
    put('front',  1, 1)
    put('right',  1, 2)
    put('back',   1, 3)
    put('bottom', 2, 1)
    return canvas


def parse_args():
    ap = argparse.ArgumentParser(description='ERP -> CubeMap converter')
    ap.add_argument('input', help='输入 ERP 图片路径（宽高比约 2:1）')
    ap.add_argument('--size', type=int, default=1024, help='每个立方体面的尺寸（像素），默认 1024')
    ap.add_argument('--out', default='out', help='输出目录（六图模式）或输出文件（十字模式）')
    ap.add_argument('--layout', choices=['six', 'cross'], default='six',
                    help='输出布局: six=六张面, cross=十字拼图')
    ap.add_argument('--interp', choices=['linear','nearest','cubic','lanczos'], default='lanczos',
                    help='重采样插值方式')
    ap.add_argument('--border', choices=['wrap','reflect','constant'], default='wrap',
                    help='经度边界处理（wrap推荐），纬度超界会按选项处理')
    return ap.parse_args()


def main():
    args = parse_args()

    interp_map = {
        'nearest': cv2.INTER_NEAREST,
        'linear' : cv2.INTER_LINEAR,
        'cubic'  : cv2.INTER_CUBIC,
        'lanczos': cv2.INTER_LANCZOS4,
    }
    border_map = {
        'wrap'    : cv2.BORDER_WRAP,
        'reflect' : cv2.BORDER_REFLECT_101,
        'constant': cv2.BORDER_CONSTANT,
    }

    erp = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if erp is None:
        raise SystemExit(f'读取失败: {args.input}')
    H, W = erp.shape[:2]
    if abs((W / max(1,H)) - 2.0) > 0.2:
        print(f'警告: 输入宽高比看起来不是 2:1（实际 {W}:{H}），请确认这是 ERP 全景图。')

    faces_order = ['right','left','top','bottom','front','back']
    maps = {}
    for name in faces_order:
        maps[name] = build_face_map(args.size, name)

    faces = {}
    for name in faces_order:
        map_x_norm, map_y_norm = maps[name]
        face_img = remap_face(erp, map_x_norm, map_y_norm,
                              interp=interp_map[args.interp],
                              border=border_map[args.border])
        faces[name] = face_img

    if args.layout == 'six':
        base = os.path.splitext(os.path.basename(args.input))[0]
        save_six_faces(faces, args.out, base)
        print(f'已输出六个面的图片到目录: {args.out}\n面名称: {faces_order}')
    else:
        cross = make_cross_layout(faces, args.size)
        ok = cv2.imwrite(args.out, cross)
        if not ok:
            raise SystemExit(f'写入失败: {args.out}')
        print(f'已输出十字拼图: {args.out}\n面名称(布局行列见代码注释): {faces_order}')


if __name__ == '__main__':
    main()
