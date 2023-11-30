from pytorch_msssim import MS_SSIM, SSIM
import torch.nn as nn
from skimage.transform import hough_line
from skimage.feature import canny
import torch 
import numpy as np

class MS_SSIM_loss(MS_SSIM):
    def forward(self, img1, img2):
        return 1 - super(MS_SSIM_loss, self).forward(img1, img2)

class SSIM_loss(SSIM):
    def forward(self, img1, img2):
        return 1 - super(SSIM_loss, self).forward(img1, img2)

class HoughTransformSSIM_loss(nn.L1Loss):
    def forward(self, pred, target):
        pred_hough = []
        target_hough = []

        for img1, img2 in zip(pred, target):
            img1 = img1.detach().numpy()
            img2 = img2.detach().numpy()
            print(img1.shape, img2.shape)
            edges1 = [canny(img) for img in img1]
            edges2 = [canny(img) for img in img2]
            h1 = []
            for edge in edges1:
                h, _, _ = hough_line(edge)
                h1.append(h.astype(np.int64))

            h2 = []
            for edge in edges2:
                h, _, _ = hough_line(edge)
                print(np.max(h), np.min(h))
                print(np.unique(h))
                h2.append(h.astype(np.int64))
            h1 = np.array(h1)
            h2 = np.array(h2)
            h1 = torch.tensor(h1)
            h2 = torch.tensor(h2)
            pred_hough.append(h1)
            target_hough.append(h2)

        # convert to int64
        pred_hough = torch.stack(pred_hough).to(dtype=torch.float32)
        target_hough = torch.stack(target_hough).to(dtype=torch.float32)
        return super(HoughTransformSSIM_loss, self).forward(pred_hough, target_hough)