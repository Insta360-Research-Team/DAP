import os
import cv2
import torch
import lightning.pytorch as pl 
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
import numpy as np
from functools import partial
from PIL import Image
from datasets import load_dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset, Subset

from src.data import RandomYawRotation

image_column = "image"
caption_column = "caption"

def get_train_dataset(file_path, remove_error_edge=False):
    dataset = load_dataset("json", data_files=file_path)
    dataset = dataset.flatten_indices()
    column_names = dataset["train"].column_names

    if image_column not in column_names:
        raise ValueError(
            f"'{image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
        )

    if caption_column not in column_names:
        raise ValueError(
            f"'{caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
        )

    return dataset


def preprocess_mixed_stage_train(examples, resolution, data_type):
    # process image
    if data_type == "perspective":
        image_transforms = transforms.Compose(
            [
                transforms.Resize((resolution, 2 * resolution),
                                interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
    elif data_type == "panorama":
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

    # process caption
    is_caption_list = isinstance(examples[caption_column][0], list)
    if is_caption_list:
        examples["captions"] = [max(example, key=len)
                                for example in examples[caption_column]]
    else:
        examples["captions"] = list(examples[caption_column])

    # process mask
    masks = []
    target_size = (resolution // 8, resolution // 4)
    for mask_path in examples["mask"]:
        mask_image = Image.open(mask_path).convert("L")
        resized_mask = mask_image.resize(target_size, Image.NEAREST)
        binary_mask = np.array(resized_mask) > 128
        masks.append(torch.from_numpy(binary_mask).bool())

    examples["masks"] = masks

    examples = {
        'pixel_values': examples['pixel_values'],
        'masks': examples['masks'], 
        'captions': examples['captions']
    }

    return examples

def prepare_mix_staged_dataset(dataset, resolution, data_type):
    dataset = dataset.with_transform(partial(preprocess_mixed_stage_train, resolution=resolution, data_type=data_type))
    return dataset

class DynamicRatioDataModule(pl.LightningDataModule):
    def __init__(self, dataset1, dataset2, resolution, batch_size, num_workers, collate_fn=None):
        super().__init__()
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.resolution = resolution
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.current_epoch = 0

    def prepare_data(self):
        pass
        
    def setup(self, stage=None):
        self.processed_dataset1 = prepare_mix_staged_dataset(
            self.dataset1, 
            self.resolution,
            data_type="panorama"
        )
        self.processed_dataset2 = prepare_mix_staged_dataset(
            self.dataset2, 
            self.resolution,
            data_type="perspective"
        )

        print(f"[Setup] Dataset1 size: {len(self.processed_dataset1)}")
        print(f"[Setup] Dataset2 size: {len(self.processed_dataset2)}")

    def train_dataloader(self):
        epoch = self.trainer.current_epoch
        print(f"Refresh Dataloader at epoch {epoch}")

        subset1 = self.processed_dataset1

        if epoch == 0 or epoch == 1:
            combined_dataset = subset1
        else:
            effective_dataset2_size = int(len(self.processed_dataset2) * (0.50 ** (epoch-1)))
            effective_dataset2_size = max(effective_dataset2_size, 1) 

            indices2 = torch.randperm(len(self.processed_dataset2))[:effective_dataset2_size]
            subset2 = Subset(self.processed_dataset2, indices2.tolist())
            
            combined_dataset = ConcatDataset([subset1, subset2])
        
        return DataLoader(
            combined_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            drop_last=True
        )