from typing import List

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms
from PIL import Image

from transformers import (
    AutoModel, 
    AutoImageProcessor,
)

from .data import ImageDataset, ImageDataloader

class DinoScorer(nn.Module):
    def __init__(
            self, 
            dino_model: str, 
            ref_images: List[str],
            ):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(dino_model)
        self.model = AutoModel.from_pretrained(dino_model)
        
        self.model.eval()
        self.model.to('cuda')
        ref_images_dataset = ImageDataset(
            images=[Image.open(img) for img in ref_images]
        )
        self.ref_images_embedding = self.get_images_embedding(ref_images_dataset)

    def get_images_embedding(self, images_dataset):
        data_loader = ImageDataloader(images_dataset, self.processor)
        embeddings = []
        with torch.no_grad():
            for batch in data_loader:
                outputs = self.model(**batch['inputs'].to('cuda'))
                embeddings.append(outputs.last_hidden_state[:, 0]) # CLS token embedding
        return torch.cat(embeddings)

    def forward(self, images):
        if isinstance(images[0], str):
            images_dataset = ImageDataset(
                images=[Image.open(img) for img in images]
            )
        elif isinstance(images[0], Image.Image):
            images_dataset = ImageDataset(images=images)
        else:
            raise ValueError('images must be a list of str or Image.Image')

        embeddings = self.get_images_embedding(images_dataset)

        # compute all-pairs cosine similarity between ref images and generated images
        # shape: (num_ref_images, num_generated_images)
        sim = F.cosine_similarity(embeddings.unsqueeze(0), self.ref_images_embedding.unsqueeze(1), dim=-1)

        # compute the mean of cosine similarity
        return sim.mean().cpu().numpy()
    
class FIDScorer(nn.Module):
    def __init__(
        self, 
        ref_images: List[str],
        feature=192, 
        ):
        super().__init__()
        self.fid_model = FrechetInceptionDistance(feature=feature, reset_real_features=False)
        self.ref_images = ref_images

        self.transform = transforms.Compose([
            transforms.PILToTensor(),
            transforms.Resize(256, interpolation=3, antialias=True),
            transforms.CenterCrop(224),
        ])
        self.__post_init__()
    
    def __post_init__(self):
        ref_imgs_tensor = self.img_to_tensor(self.ref_images)
        self.fid_model.update(ref_imgs_tensor, real=True)

    def img_to_tensor(self, imgs):
        imgs_list = []
        for img in imgs:
            if isinstance(img, str):
                img = Image.open(img)
            elif isinstance(img, Image.Image):
                pass
            else:
                raise ValueError('images must be a list of str or Image.Image')
            img = self.transform(img)
            imgs_list.append(img)
        return torch.stack(imgs_list)
        
    def forward(self, images):
        imgs_tensor = self.img_to_tensor(images)
        self.fid_model.update(imgs_tensor, real=False)
        fid = self.fid_model.compute()
        self.fid_model.reset()
        return fid.item()