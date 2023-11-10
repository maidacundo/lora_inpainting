from torch.utils.data import Dataset
from PIL import Image
import random
from pathlib import Path
from typing import Optional
from roboflow import Roboflow
import os
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms
from torchvision.transforms.v2 import functional as TF

import numpy as np
import cv2

from .utils import parse_labels, scale_polygons, generate_masks, mask_image, get_images_and_labels_paths

class InpaintLoraDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        tokenizer,
        label_mapping: dict,
        global_caption: Optional[str] = None,
        size=512,
        normalize=True,
        augmentation=True,
        scaling_pixels: int = 0,
        labels_filter: Optional[list] = None,
        to_tensor: bool = True,
    ):
        self.size = size
        self.tokenizer = tokenizer
        self.augmentation = augmentation

        if not Path(instance_data_root).exists():
            raise ValueError("Instance images root doesn't exists.")

        # Prepare the instance images and masks
        self.imgs, self.labels = get_images_and_labels_paths(instance_data_root)
        self.label_mapping = label_mapping

        self.global_caption = global_caption

        self._length = len(self.imgs)

        self.normalize = normalize
        self.scaling_pixels = scaling_pixels
        self.labels_filter = labels_filter # TODO implement labels filter
        self.to_tensor = to_tensor

        self.mean, self.std = self.calculate_mean_std()

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size=self.size),
                transforms.ToImageTensor()
                if self.to_tensor
                else transforms.Lambda(lambda x: x),
                transforms.ConvertImageDtype(torch.float32)
                if self.to_tensor
                else transforms.Lambda(lambda x: x),
                transforms.Normalize(mean=self.mean, std=self.std)
                if self.normalize
                else transforms.Lambda(lambda x: x),
            ]
        )

        self.mask_transforms = transforms.Compose(
            [
                transforms.Resize(size=self.size),
                transforms.PILToTensor(),
            ]
        )

    def calculate_mean_std(self):
        means = []
        stds = []

        for img in self.imgs:
            # calculate the mean and std of all the images
            img = cv2.imread(img, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0
            mean = np.mean(img, axis=(0, 1))
            std = np.std(img, axis=(0, 1))
            means.append(mean)
            stds.append(std)

        mean = np.mean(means, axis=0)
        std = np.mean(stds, axis=0)

        return mean, std

    def transform(self, image, mask):
        
        mask = torch.from_numpy(mask).unsqueeze(0)

        image = self.image_transforms(image)
        mask = self.mask_transforms(mask)

        if self.augmentation:

            while True:
                # Random crop
                i, j, h, w = transforms.RandomCrop.get_params(
                    image, output_size=(self.size, self.size))
                image = TF.crop(image, i, j, h, w)
                mask = TF.crop(mask, i, j, h, w)

                # Check if the mask contains at least one non-zero value
                # TODO find a bug in the mask generation
                # because sometimes the mask is all zeros
                # when the size is not 512 (e.g. 768)
                if torch.sum(mask) > 0:
                    break


            # Random horizontal flipping
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

        return image, mask
    
    
    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}

        image = Image.open(self.imgs[index])
        polygons = parse_labels(self.labels[index])
        scaled_polygons = scale_polygons(polygons, image.size)
        mask, label = generate_masks(
            scaled_polygons, 
            image.size, 
            scaling_pixels=self.scaling_pixels
        )
        image, mask = self.transform(image, mask)


        example["instance_images"] = image
        example["instance_masks"] = mask
        example["instance_masked_images"] = mask_image(image, example["instance_masks"], invert=True)
        example["instance_masked_values"] = mask_image(image, example["instance_masks"], invert=False)

        text = self.label_mapping[label]
        if self.global_caption:
            text += ', ' + self.global_caption.strip()

        example["instance_prompt_ids"] = self.tokenizer(
            text,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        return example

class InpaintingDataLoader(DataLoader):
    def __init__(self, dataset, tokenizer, device, batch_size=1, **kwargs):
        super().__init__(dataset, collate_fn=self.collate_fn, batch_size=batch_size, **kwargs)
        self.tokenizer = tokenizer
        self.device = device

    def collate_fn(self, examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]
        mask_values = [example["instance_masked_values"] for example in examples]
        masked_image_values = [
            example["instance_masked_images"] for example in examples
        ]
        mask = [example["instance_masks"] for example in examples]

        # Concat class and instance examples for prior preservation.
        # We do this to avoid doing two forward passes.
        if examples[0].get("class_prompt_ids", None) is not None:
            input_ids += [example["class_prompt_ids"] for example in examples]
            pixel_values += [example["class_images"] for example in examples]
            mask_values += [example["class_masks"] for example in examples]
            masked_image_values += [
                example["class_masked_images"] for example in examples
            ]

        pixel_values = (
            torch.stack(pixel_values).to(memory_format=torch.contiguous_format).float()
        )
        mask_values = (
            torch.stack(mask_values).to(memory_format=torch.contiguous_format).float()
        )
        masked_image_values = (
            torch.stack(masked_image_values).to(memory_format=torch.contiguous_format).float()
        )
        mask = (
            torch.stack(mask).to(memory_format=torch.contiguous_format).float()
        )

        input_ids = self.tokenizer.pad(
            {"input_ids": input_ids},
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "mask_values": mask_values,
            "masked_image_values": masked_image_values,
            "mask": mask,
        }

        return batch

def download_roboflow_dataset(config):
    rf = Roboflow(api_key=config.dataset.roboflow_api_key)
    project = rf.workspace(config.dataset.roboflow_workspace).project(config.dataset.project_name)
    dataset = project.version(config.dataset.dataset_version).download("yolov7", location=config.dataset.data_root)