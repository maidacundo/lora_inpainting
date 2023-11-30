from pytorch_msssim import MS_SSIM, SSIM
import torch.nn as nn
import torch 
from controlnet_aux import MLSDdetector

class MS_SSIM_loss(MS_SSIM):
    def forward(self, img1, img2):
        return 1 - super(MS_SSIM_loss, self).forward(img1, img2)

class SSIM_loss(SSIM):
    def forward(self, img1, img2):
        return 1 - super(SSIM_loss, self).forward(img1, img2)

class HoughTransform_loss(nn.L1Loss):
    def __init__(self, *args, **kwargs):
        super(HoughTransform_loss, self).__init__(*args, **kwargs)
        self.mlsd = MLSDdetector.from_pretrained('lllyasviel/ControlNet').model

    def forward(self, pred, target):
        mlsd_out = self.mlsd(torch.cat([pred, target], dim=0))
        pred_hough, target_hough = mlsd_out.chunk(2, dim=0)
        return super(HoughTransform_loss, self).forward(pred_hough, target_hough)