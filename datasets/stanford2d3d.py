from __future__ import print_function
import os
import cv2
import numpy as np
import random

import torch
from torch.utils import data
from torchvision import transforms

from .util import Equirec2Cube


def read_list(list_file):
    rgb_depth_list = []
    with open(list_file) as f:
        lines = f.readlines()
        for line in lines:
            rgb_depth_list.append(line.strip().split(" "))
    return rgb_depth_list


class Stanford2D3D(data.Dataset):
    """The Stanford2D3D Dataset"""

    def __init__(self, root_dir, list_file, height=512, width=1024, color_augmentation=True,
                 LR_filp_augmentation=True, yaw_rotation_augmentation=True, repeat=1, is_training=False):
        """
        Args:
            root_dir (string): Directory of the Stanford2D3D Dataset.
            list_file (string): Path to the txt file contain the list of image and depth files.
            height, width: input size.
            disable_color_augmentation, disable_LR_filp_augmentation,
            disable_yaw_rotation_augmentation: augmentation options.
            is_training (bool): True if the dataset is the training set.
        """
        self.root_dir = root_dir
        self.rgb_depth_list = read_list(list_file)
        if is_training:
            self.rgb_depth_list = 10 * self.rgb_depth_list
        self.w = width
        self.h = height

        self.max_depth_meters = 100.0

        self.color_augmentation = color_augmentation
        self.LR_filp_augmentation = LR_filp_augmentation
        self.yaw_rotation_augmentation = yaw_rotation_augmentation

        self.is_training = is_training

        self.e2c = Equirec2Cube(self.h, self.w, self.h // 2)

        if self.color_augmentation:
            try:
                self.brightness = (0.8, 1.2)
                self.contrast = (0.8, 1.2)
                self.saturation = (0.8, 1.2)
                self.hue = (-0.1, 0.1)
                self.color_aug= transforms.ColorJitter(
                    self.brightness, self.contrast, self.saturation, self.hue)
            except TypeError:
                self.brightness = 0.2
                self.contrast = 0.2
                self.saturation = 0.2
                self.hue = 0.1
                self.color_aug = transforms.ColorJitter(
                    self.brightness, self.contrast, self.saturation, self.hue)

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.rgb_depth_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        inputs = {}

        rgb_name = os.path.join(self.root_dir, self.rgb_depth_list[idx][0])
        # rgb_name = rgb_name.replace("rgb_", "rgb_pre_") # use the preprocessed rgb images
        rgb = cv2.imread(rgb_name)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, dsize=(self.w, self.h), interpolation=cv2.INTER_CUBIC)

        depth_name = os.path.join(self.root_dir, self.rgb_depth_list[idx][1])
        gt_depth = cv2.imread(depth_name, -1)
        gt_depth = cv2.resize(gt_depth, dsize=(self.w, self.h), interpolation=cv2.INTER_NEAREST)
        gt_depth = gt_depth.astype(float) / 512
        gt_depth[gt_depth > self.max_depth_meters+1] = self.max_depth_meters + 1

        # gt_depth = (gt_depth - gt_depth.min()) / (gt_depth.max() - gt_depth.min())
        # gt_depth = (gt_depth * 255).astype(np.uint8)
        # vis_img =  np.concatenate([rgb, cv2.cvtColor(gt_depth, cv2.COLOR_GRAY2BGR)], axis=0)
        # cv2.imwrite(f"./shit/vis_img_{idx}.png", vis_img)

        if self.is_training and self.yaw_rotation_augmentation:
            # random yaw rotation
            roll_idx = random.randint(0, self.w)
            rgb = np.roll(rgb, roll_idx, 1)
            gt_depth = np.roll(gt_depth, roll_idx, 1)

        if self.is_training and self.LR_filp_augmentation and random.random() > 0.5:
            rgb = cv2.flip(rgb, 1)
            gt_depth = cv2.flip(gt_depth, 1)

        if self.is_training and self.color_augmentation and random.random() > 0.5:
            aug_rgb = np.asarray(self.color_aug(transforms.ToPILImage()(rgb)))
        else:
            aug_rgb = rgb

        cube_rgb = self.e2c.run(rgb)
        cube_aug_rgb = self.e2c.run(aug_rgb)

        rgb = self.to_tensor(rgb.copy())
        cube_rgb = self.to_tensor(cube_rgb.copy())
        aug_rgb = self.to_tensor(aug_rgb.copy())
        cube_aug_rgb = self.to_tensor(cube_aug_rgb.copy())
        gt_depth = torch.from_numpy(np.expand_dims(gt_depth, axis=0)).to(torch.float32)

        val_mask = ((gt_depth > 0) & (gt_depth <= self.max_depth_meters) & ~torch.isnan(gt_depth))

        # 归一化深度值，0.01-1.0
        # _min, _max = torch.quantile(gt_depth[val_mask], torch.tensor([0.02, 1 - 0.02]),)
        gt_depth_norm = gt_depth / self.max_depth_meters
        gt_depth_norm = torch.clip(gt_depth_norm, 0.001, 1.0)


        inputs["rgb"] = self.normalize(aug_rgb)
        inputs["normalized_rgb"] = self.normalize(aug_rgb)

        inputs["cube_rgb"] = cube_rgb
        inputs["normalized_cube_rgb"] = self.normalize(cube_aug_rgb)

        inputs["gt_depth"] = gt_depth_norm
        inputs["val_mask"] = val_mask # 合法区域，不是全true，真把不能用的区域划出来了；其他参与训练的数据集是全true的（除了投影数据集）
        inputs["mask_100"] = (gt_depth > 0) & (gt_depth <= 100) 
        # 对于这个数据集，mask_100设定为全true的，因为求不出来。大于100米的深度gt也有可能是玻璃镜子等物体，反正这个数据集也不参加训练
        
        # 这个数据集中，模型预测的mask100应该是被val_mask涵盖的，所以mask100理论上没有影响
        # val_mask控制计算指标的区域
        return inputs



