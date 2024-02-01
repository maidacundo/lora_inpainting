from controlnet_aux import MLSDdetector
from pytorch_msssim import MS_SSIM, SSIM
import torch.nn as nn
import torch

class MS_SSIM_loss(MS_SSIM):
    def forward(self, img1, img2):
        return 1 - super(MS_SSIM_loss, self).forward(img1, img2)

class SSIM_loss(SSIM):
    def forward(self, img1, img2):
        return 1 - super(SSIM_loss, self).forward(img1, img2)
    
class MLSD_Perceptual_loss(nn.L1Loss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mlsd = MLSDdetector.from_pretrained('lllyasviel/ControlNet').model
        self.mlsd.eval()
        self.mlsd.requires_grad_(False)
        self.mlsd.to('cuda')
    
    def forward(self, step_latents, original_latents):
        mlsd_features = self.mlsd(torch.cat([step_latents, original_latents], dim=0))
        pred_features, target_features = mlsd_features.chunk(2, dim=0)
        return super().forward(pred_features, target_features)

