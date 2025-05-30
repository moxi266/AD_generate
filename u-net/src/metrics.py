import os
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from skimage.metrics import (
    structural_similarity as ssim,
    peak_signal_noise_ratio as psnr,
)

import torch
import SimpleITK as sitk
import numpy as np
from monai.transforms import CenterSpatialCrop, SpatialPad
import torch.nn.functional as F


def norm(img):
    min_val = img.min()
    max_val = img.max()
    # 将张量的值缩放到 [0, 1]
    normalized_tensor = (img - min_val) / (max_val - min_val)
    return normalized_tensor


def MAE(true, pre):
    return torch.mean(torch.abs(true - pre))


def MSE(true, pre):
    return torch.mean((true - pre) ** 2)

def RMSE(true, pre):
    return torch.sqrt(torch.mean((true - pre) ** 2))


def SSIM(true, pre):
    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(
        data_range=1.0, kernel_size=7
    )  # √
    ssim = ms_ssim(true, pre)
    return ssim


def PSNR(true, pre):
    return 10 * torch.log10(1 / MSE(true, pre))


if __name__ == "__main__":

    img1 = norm(torch.randn([10, 1,128, 128]))
    img2 = norm(torch.randn([10, 1,128, 128]))

    mae = MAE(img1, img2)
    ssim = SSIM(img1, img2)
    psnr = PSNR(img1, img2)
    print("mae: {}, ssim: {}, psnr: {}".format(mae, ssim, psnr))
