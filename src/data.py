from torch.utils.data import Dataset
from PIL import Image
import os
import json
import random
import torch
import cv2
import numpy as np
from PIL import Image
from functools import partial
from transformers import AutoImageProcessor, AutoProcessor, CLIPProcessor
from datasets import load_dataset
from torchvision import transforms


class RandomYawRotation:
    """
    Apply a random yaw rotation to an equirectangular (panoramic) image.

    For equirectangular projections, a pure yaw rotation in 3D space
    is mathematically equivalent to a horizontal circular shift in the image domain.
    This transform simulates viewpoint changes efficiently without distorting geometry.
    """

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Args:
            img (PIL.Image.Image): Input equirectangular image.

        Returns:
            PIL.Image.Image: The image after a random horizontal shift.
        """
        img_np = np.array(img)
        shift = random.randint(img_np.shape[1] // 3, img_np.shape[1])
        rolled_img_np = np.roll(img_np, shift, axis=1)
        return Image.fromarray(rolled_img_np)


def preprocess_train(examples, resolution):

    image_transforms = transforms.Compose(
        [
            transforms.Resize((resolution, 2 * resolution),
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5), 
            RandomYawRotation(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    images = [(image.convert("RGB") if not isinstance(image, str) else Image.open(
        image).convert("RGB")) for image in examples["image"]]
    images = [image_transforms(image) for image in images]

    examples["pixel_values"] = images
    examples["captions"] = list(examples["caption"])

    return examples


def prepare_train_dataset(dataset, resolution):

    dataset = dataset['train'].with_transform(
        partial(preprocess_train, resolution=resolution))

    return dataset
