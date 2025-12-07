import torch
import numpy as np
from PIL import Image
import cv2
from typing import Optional


def shift_tensor(tensor, x):
    shifted_tensor = torch.zeros_like(tensor)

    if x > 0:
        shifted_tensor[:, x:] = tensor[:, :-x]
    elif x < 0:
        shifted_tensor[:, :x] = tensor[:, -x:]
    else:
        shifted_tensor = tensor  # No shift for x == 0

    return shifted_tensor


def create_mask(input_image_path, w=64, h=64):
    img = (
        Image.open(input_image_path)
        .resize((w, h), Image.Resampling.NEAREST)
        .convert("L")
    )
    img_array = np.array(img)
    mask = np.where(img_array == 255, 1, 0)
    mask_tensor = torch.tensor(mask).int()

    return mask_tensor


def save_array_as_png(array, path):
    if array.dtype != np.uint8:
        array = (array * 255).clip(0, 255).astype(np.uint8)
    image = Image.fromarray(array, "RGBA")
    image.save(path)


def convert_to_mask_inpainting(image_array, mask_path):
    if image_array.shape[2] != 4:
        raise ValueError("输入数组必须是 RGBA 格式")
    mask = np.ones(image_array.shape[:2], dtype=np.uint8) * 255
    alpha_channel = image_array[:, :, 3]
    mask[alpha_channel != 0] = 0
    mask_image = Image.fromarray(mask, mode="L")
    mask_image.save(mask_path)

    return mask_image


# mask for Subject Customiztion
def composite_images(background_path: str, mask_path: str) -> Image.Image:
    background = Image.open(background_path).convert("RGBA")
    mask = Image.open(mask_path).convert("L")

    if background.size != mask.size:
        mask = mask.resize(background.size)

    mask_array = np.array(mask) > 128

    if background.mode == "RGBA":
        white_canvas = Image.new("RGBA", background.size, (255, 255, 255, 255))
    else:
        white_canvas = Image.new("RGB", background.size, (255, 255, 255))

    composite = Image.composite(background, white_canvas, Image.fromarray(mask_array))

    return composite.convert("RGB")


def process_mask_array(mask_array: np.ndarray) -> Image.Image:
    alpha = mask_array[..., 3]
    gray_array = np.where(alpha > 0, 0, 255).astype(np.uint8)
    mask_image = Image.fromarray(gray_array, mode="L")
    return mask_image.convert("1")


def process_mask(mask: Image.Image) -> Image.Image:
    if mask.mode != "L":
        mask = mask.convert("L")
    return mask.point(lambda x: 1 if x > 128 else 0, mode="1")


def merge_masks(mask1: Image.Image, mask2: Image.Image) -> Image.Image:
    arr1 = np.array(mask1, dtype=bool)
    arr2 = np.array(mask2, dtype=bool)
    merged = np.logical_and(arr1, arr2)
    return Image.fromarray(merged).convert("1")


def save_merged_mask(
    mask_array: np.ndarray, mask: Optional[Image.Image], output_path: str
) -> None:
    mask1 = process_mask_array(mask_array)

    if mask is not None:
        mask2 = process_mask(mask)
        if mask1.size != mask2.size:
            mask2 = mask2.resize(mask1.size, Image.NEAREST)
        merged = merge_masks(mask1, mask2)
    else:
        merged = mask1

    merged.save(output_path)
