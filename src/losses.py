from pytorch_msssim import MS_SSIM, SSIM
import torch.nn as nn
import torch

class MS_SSIM_loss(MS_SSIM):
    def forward(self, img1, img2):
        return 1 - super(MS_SSIM_loss, self).forward(img1, img2)

class SSIM_loss(SSIM):
    def forward(self, img1, img2):
        return 1 - super(SSIM_loss, self).forward(img1, img2)


