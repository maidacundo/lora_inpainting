from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random
from pathlib import Path
from typing import Optional
from roboflow import Roboflow
import os
import torch
from torch.utils.data import DataLoader

from .utils import mask_images, create_dataset

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
        token_map: Optional[dict] = None,
        size=512,
        max_size=628,
        h_flip=True,
        resize=True,
        normalize=True,
        scaling_pixels: int = 0,
        train_inpainting: bool = True,
        labels_filter: Optional[list]=None,
    ):
        self.size = size
        self.max_size = max_size
        self.tokenizer = tokenizer
        self.resize = resize
        self.train_inpainting = train_inpainting

        if not Path(instance_data_root).exists():
            raise ValueError("Instance images root doesn't exists.")
        img_path = os.path.join(instance_data_root, "images")
        label_path = os.path.join(instance_data_root, "labels")

        # Prepare the instance images and masks
        self.imgs, self.masks, self.labels = create_dataset(img_path, label_path, num_samples=None, scaling_pixels=scaling_pixels, labels_filter=labels_filter)
        self.label_mapping = label_mapping

        self.global_caption = global_caption

        self.token_map = token_map

        self._length = len(self.imgs)

        self.h_flip = h_flip
        self.image_transforms = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(
                    size, 
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    max_size=max_size,
                )
                if resize
                else transforms.Lambda(lambda x: x),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
                if normalize
                else transforms.Lambda(lambda x: x),
            ]
        )
        self.mask_transforms = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(
                    size, 
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    max_size=max_size,
                )
                if resize
                else transforms.Lambda(lambda x: x),
                transforms.CenterCrop(size),
                transforms.PILToTensor()
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        label = self.labels[index]

        if self.train_inpainting:
            example["instance_masks"] = self.masks[index]
            example["instance_masked_images"] = mask_images(self.imgs[index], example["instance_masks"], invert=True)
            example["instance_masked_values"] = mask_images(self.imgs[index], example["instance_masks"], invert=False)

        example["instance_images"] = self.image_transforms(self.imgs[index])
        example["instance_masked_images"] = self.image_transforms(example["instance_masked_images"])
        example["instance_masked_values"] = self.image_transforms(example["instance_masked_values"])
        example["instance_masks"] = self.mask_transforms(example["instance_masks"])

        text = self.label_mapping[label]
        if self.global_caption:
            text += ', ' + self.global_caption.strip()

        if self.h_flip and random.random() > 0.5:
            hflip = transforms.RandomHorizontalFlip(p=1)

            example["instance_images"] = hflip(example["instance_images"])
            example["instance_masked_images"] = hflip(example["instance_masked_images"])
            example["instance_masked_values"] = hflip(example["instance_masked_values"])
            example["instance_masks"] = hflip(example["instance_masks"])

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