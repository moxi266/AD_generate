import os
import sys
import argparse
import warnings
import pandas as pd
import torch
import torch.nn as nn
import datetime
from tqdm import tqdm
from monai import transforms
from monai.utils import set_determinism
from generative.losses import PerceptualLoss, PatchAdversarialLoss
from torch.nn import L1Loss
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import numpy as np
import nibabel as nib

sys.path.append("/staff/limingchao/AD/mycode")
from src import const
from src import utils
from src import (
    KLDivergenceLoss,
    GradientAccumulation,
    init_autoencoder,
    init_patch_discriminator,
    get_dataset_from_pd,
    UNet3D_Generator,
    init_AttentionUnet,
    init_Unet,
)

from src.metrics import MAE, SSIM, PSNR, RMSE


set_determinism(3407)
device = "cuda:7" if torch.cuda.is_available() else "cpu"


def test_model(args):
    # 定义数据转换，与训练时一致
    transforms_fn = transforms.Compose(
        [
            transforms.CopyItemsD(keys={"MRI_path"}, names=["mri"]),
            transforms.CopyItemsD(keys={"PET_path"}, names=["pet"]),
            transforms.LoadImageD(image_only=True, keys=["mri", "pet"]),
            transforms.EnsureChannelFirstD(keys=["mri", "pet"]),
            transforms.SpacingD(pixdim=const.RESOLUTION, keys=["mri", "pet"]),
            transforms.ResizeWithPadOrCropD(
                spatial_size=(120, 144, 120),
                mode="minimum",
                keys=["mri", "pet"],
            ),
            transforms.ScaleIntensityD(minv=0, maxv=1, keys=["mri", "pet"]),
        ]
    )

    # 加载测试数据
    dataset_df = pd.read_csv(args.dataset_csv)
    test_df = dataset_df[dataset_df.split == "test"]
    testset = get_dataset_from_pd(test_df, transforms_fn, args.cache_dir)
    test_loader = DataLoader(
        dataset=testset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=False,
        persistent_workers=True,
        pin_memory=True,
    )

    # 加载训练好的模型
    G = init_Unet(args.G_ckpt).to(device)  # MRI → PET

    # 定义指标累加器
    mae_list, ssim_list, psnr_list, rmse_list = [], [], [], []
    MNI152_1P5MM_AFFINE = np.array(
        [
            [-1.5, 0.0, 0.0, 90.0],
            [0.0, 1.5, 0.0, -126.0],
            [0.0, 0.0, 1.5, -72.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    synthesis_path = os.path.join(args.synthesis_dir, args.pet)
    if not os.path.exists(synthesis_path):
        os.makedirs(synthesis_path)
    txt_path = os.path.join(synthesis_path, "u-net_results.txt")
    # 测试循环
    G.eval()
    with torch.no_grad():
        progress_bar = tqdm(
            enumerate(test_loader), total=len(test_loader), desc="Testing"
        )
        for step, batch in progress_bar:
            with autocast(device_type="cuda", enabled=True):
                mri_id = batch["Image Data ID_MRI"][0]
                subject = batch["Subject"][0]
                pet_id = batch["Image Data ID_PET"][0]
                Diagnosis = batch["Group"][0]
                sex = batch["Sex"][0]
                age = batch["Age"][0]

                pet_gen = G(batch["mri"].to(device))

                # 真实PET图像读取
                pet_true = batch["pet"].to(device)

                # MAE、SSIM、PSNR 函数接受形状为 [1, D, H, W] 的 tensor
                pet_gen = pet_gen.detach().cpu().float()
                pet_true = pet_true.detach().cpu().float()
                mae_val = MAE(pet_gen, pet_true)
                ssim_val = SSIM(pet_gen, pet_true)
                psnr_val = PSNR(pet_gen, pet_true)
                rmse_val = RMSE(pet_gen, pet_true)

                mae_list.append(mae_val)
                ssim_list.append(ssim_val)
                psnr_list.append(psnr_val)
                rmse_list.append(rmse_val)
                if args.synthesis:
                    # 去掉batch维度和通道维度
                    pet_gen = pet_gen.squeeze(0).squeeze(0).numpy()[:, ::-1, :]
                    pet_true = pet_true.squeeze(0).squeeze(0).numpy()[:, ::-1, :]
                    gen_img = nib.Nifti1Image(pet_gen, affine=MNI152_1P5MM_AFFINE)
                    true_img = nib.Nifti1Image(pet_true, affine=MNI152_1P5MM_AFFINE)
                    gen_path = os.path.join(
                        synthesis_path, f"{subject}_{pet_id}_{mri_id}_u-net.nii.gz"
                    )
                    # true_path = os.path.join(
                    #     synthesis_path, f"{subject}_{pet_id}_{mri_id}.nii.gz"
                    # )
                    gen_img.to_filename(gen_path)
                    # true_img.to_filename(true_path)
                    info_str = (
                        f"subject: {subject}    mri_id: {mri_id}    pet_id: {pet_id}    "
                        f"age: {age}    sex: {sex}    diagnosis: {Diagnosis}\n"
                        f"mae_val: {mae_val:.4f}\n"
                        f"rmse_val: {rmse_val:.4f}\n"
                        f"psnr_val: {psnr_val:.4f}\n"
                        f"ssim_val: {ssim_val:.4f}\n\n"
                    )
                    # 写入到txt文件（追加模式）
                    with open(txt_path, "a") as f:
                        f.write(info_str)
                progress_bar.set_postfix(
                    {
                        "MAE": mae_val,
                        "RMSE": rmse_val,
                        "PSNR": psnr_val,
                        "SSIM": ssim_val,
                    }
                )

    avg_mae = np.mean(mae_list)
    avg_ssim = np.mean(ssim_list)
    avg_psnr = np.mean(psnr_list)
    avg_rmse = np.mean(rmse_list)
    with open(txt_path, "a") as f:
        f.write(
            f"Test Metrics: MAE={avg_mae:.4f}, RMSE={avg_rmse:.4f}, PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}"
        )
    print(
        f"Test Metrics: MAE={avg_mae:.4f}, RMSE={avg_rmse:.4f}, PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # /staff/limingchao/AD/csv/new/FDG-PET_dataset_3.csv
    # /staff/limingchao/AD/csv/new/Tau-PET_dataset_3.csv
    # /staff/limingchao/AD/csv/new/abeta-PET_dataset_3.csv
    parser.add_argument(
        "--dataset_csv",
        default="/staff/limingchao/AD/csv/last/abeta-PET_dataset_3.csv",
        type=str,
    )
    parser.add_argument(
        "--cache_dir",
        default="/staff/limingchao/AD/mycode/u-net/cache",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        default="/staff/limingchao/AD/mycode/u-net/output1/",
        type=str,
    )
    parser.add_argument(
        "--synthesis_dir",
        default="/staff/limingchao/AD/mycode/cycle_gan/synthesis1",
        type=str,
    )
    parser.add_argument("--pet", default="abeta", type=str)

    parser.add_argument(
        "--G_ckpt",
        default="/staff/limingchao/AD/mycode/u-net/output1/abeta_2025-05-09_14-13-14/best/G-ep-58.pth",
        type=str,
    )
    parser.add_argument(
        "--synthesis",
        action="store_false",
    )
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    args = parser.parse_args()

    test_model(args)
