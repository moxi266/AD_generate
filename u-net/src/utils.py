from typing import Union

import numpy as np
import nibabel as nib
import torch
import matplotlib.pyplot as plt
from nibabel.processing import resample_from_to
from monai import transforms
from monai.data.meta_tensor import MetaTensor
from torch.utils.tensorboard.writer import SummaryWriter
from torch.amp import autocast


class AverageLoss:
    """
    Utility class to track losses
    and metrics during training.
    """

    def __init__(self):
        self.losses_accumulator = {}

    def put(self, loss_key: str, loss_value: Union[int, float]) -> None:
        """
        Store value

        Args:
            loss_key (str): Metric name
            loss_value (int | float): Metric value to store
        """
        if loss_key not in self.losses_accumulator:
            self.losses_accumulator[loss_key] = []
        self.losses_accumulator[loss_key].append(loss_value)

    def pop_avg(self, loss_key: str) -> float:
        """
        Average the stored values of a given metric

        Args:
            loss_key (str): Metric name

        Returns:
            float: average of the stored values
        """
        if loss_key not in self.losses_accumulator:
            return None
        losses = self.losses_accumulator[loss_key]
        self.losses_accumulator[loss_key] = []
        return sum(losses) / len(losses)

    def to_tensorboard(self, writer: SummaryWriter, step: int):
        """
        Logs the average value of all the metrics stored
        into Tensorboard.

        Args:
            writer (SummaryWriter): Tensorboard writer
            step (int): Tensorboard logging global step
        """
        for metric_key in self.losses_accumulator.keys():
            writer.add_scalar(metric_key, self.pop_avg(metric_key), step)


def to_vae_latent_trick(
    z: torch.Tensor, unpadded_z_shape: tuple = (3, 15, 18, 15)
) -> torch.Tensor:
    """
    The latent for the VAE is not divisible by 4 (required to
    go through the UNet), therefore we apply padding before using
    it with the UNet. This function removes the padding.

    Args:
        z (torch.Tensor): Padded latent
        unpadded_z_shape (tuple, optional): unpadded latent dimensions. Defaults to (3, 15, 18, 15).

    Returns:
        torch.Tensor: Latent without padding
    """
    padder = transforms.DivisiblePad(k=4)
    z = padder(MetaTensor(torch.zeros(unpadded_z_shape))) + z
    z = padder.inverse(z)
    return z


def to_mni_space_1p5mm_trick(
    x: torch.Tensor, mni1p5_dim: tuple = (122, 146, 122)
) -> torch.Tensor:
    """
    The volume is resized to be divisible by 8 (required by
    the autoencoder). This function restores the initial dimensions
    (i.e., the MNI152 space dimensions at 1.5 mm^3).

    Args:
        x (torch.Tensor): Resized volume
        mni1p5_dim (tuple, optional): MNI152 space dims at 1.5 mm^3. Defaults to (122, 146, 122).

    Returns:
        torch.Tensor: input resized to original shape
    """
    resizer = transforms.ResizeWithPadOrCrop(spatial_size=mni1p5_dim, mode="minimum")
    return resizer(x)


def tb_display_reconstruction(writer, step, image, recon):
    """
    Display reconstruction in TensorBoard during AE training.
    """
    plt.style.use("dark_background")
    _, ax = plt.subplots(ncols=3, nrows=2, figsize=(7, 5))
    for _ax in ax.flatten():
        _ax.set_axis_off()

    if len(image.shape) == 4:
        image = image.squeeze(0)
    if len(recon.shape) == 4:
        recon = recon.squeeze(0)

    ax[0, 0].set_title("original image", color="cyan")
    ax[0, 0].imshow(image[image.shape[0] // 2, :, :], cmap="gray")
    ax[0, 1].imshow(image[:, image.shape[1] // 2, :], cmap="gray")
    ax[0, 2].imshow(image[:, :, image.shape[2] // 2], cmap="gray")

    ax[1, 0].set_title("reconstructed image", color="magenta")
    ax[1, 0].imshow(recon[recon.shape[0] // 2, :, :], cmap="gray")
    ax[1, 1].imshow(recon[:, recon.shape[1] // 2, :], cmap="gray")
    ax[1, 2].imshow(recon[:, :, recon.shape[2] // 2], cmap="gray")

    plt.tight_layout()
    writer.add_figure("Reconstruction", plt.gcf(), global_step=step)


def tb_display_generation(writer, step, tag, image):
    """
    Display generation result in TensorBoard during Diffusion Model training.
    """
    plt.style.use("dark_background")
    _, ax = plt.subplots(ncols=3, figsize=(7, 3))
    for _ax in ax.flatten():
        _ax.set_axis_off()

    ax[0].imshow(image[image.shape[0] // 2, :, :], cmap="gray")
    ax[1].imshow(image[:, image.shape[1] // 2, :], cmap="gray")
    ax[2].imshow(image[:, :, image.shape[2] // 2], cmap="gray")

    plt.tight_layout()
    writer.add_figure(tag, plt.gcf(), global_step=step)


def tb_display_cond_generation(
    writer, step, tag, starting_image, followup_image, predicted_image
):
    """
    Display conditional generation result in TensorBoard during ControlNet training.
    """
    plt.style.use("dark_background")
    _, ax = plt.subplots(ncols=3, nrows=3, figsize=(7, 7))
    for _ax in ax.flatten():
        _ax.set_axis_off()

    ax[0, 0].set_title("starting image", color="cyan")
    ax[0, 0].imshow(starting_image[starting_image.shape[0] // 2, :, :], cmap="gray")
    ax[0, 1].imshow(starting_image[:, starting_image.shape[1] // 2, :], cmap="gray")
    ax[0, 2].imshow(starting_image[:, :, starting_image.shape[2] // 2], cmap="gray")

    ax[1, 0].set_title("follow-up image", color="magenta")
    ax[1, 0].imshow(followup_image[followup_image.shape[0] // 2, :, :], cmap="gray")
    ax[1, 1].imshow(followup_image[:, followup_image.shape[1] // 2, :], cmap="gray")
    ax[1, 2].imshow(followup_image[:, :, followup_image.shape[2] // 2], cmap="gray")

    ax[2, 0].set_title("predicted follow-up", color="yellow")
    ax[2, 0].imshow(predicted_image[predicted_image.shape[0] // 2, :, :], cmap="gray")
    ax[2, 1].imshow(predicted_image[:, predicted_image.shape[1] // 2, :], cmap="gray")
    ax[2, 2].imshow(predicted_image[:, :, predicted_image.shape[2] // 2], cmap="gray")

    plt.tight_layout()
    writer.add_figure(tag, plt.gcf(), global_step=step)


def percnorm_nifti(mri, lperc=1, uperc=99):
    """
    Apply percnorm to NiFTI1Image class
    """
    norm_arr = percnorm(mri.get_fdata(), lperc, uperc)
    return nib.Nifti1Image(norm_arr, mri.affine, mri.header)


def percnorm(arr, lperc=1, uperc=99):
    """
    Remove outlier intensities from a brain component,
    similar to Tukey's fences method.
    """
    upperbound = np.percentile(arr, uperc)
    lowerbound = np.percentile(arr, lperc)
    arr[arr > upperbound] = upperbound
    arr[arr < lowerbound] = lowerbound
    return arr


def apply_mask(mri, segm):
    """
    Performs brain extraction.
    """
    segm = resample_from_to(segm, mri, order=0)
    mask = segm.get_fdata() > 0
    mri_arr = mri.get_fdata()
    mri_arr[mask == 0] = 0
    return nib.Nifti1Image(mri_arr, mri.affine, mri.header)


# 定义一个函数，根据图像尺寸获取三个视图
def get_views(img):
    axial = img[img.shape[0] // 2, :, :]  # 横断面：沿第一个维度取中间切片
    sagittal = img[:, img.shape[1] // 2, :]  # 矢状面：沿第二个维度取中间切片
    coronal = img[:, :, img.shape[2] // 2]  # 冠状面：沿第三个维度取中间切片
    return axial, sagittal, coronal


def tb_display_MRI_PET(writer, step, mri, reconstruction, pet):
    """
    在 TensorBoard 中显示跨模态图像对比：
      - 第一行：MRI 的横断面、矢状面、冠状面
      - 第二行：重建的 PET 的横断面、矢状面、冠状面
      - 第三行：真实 PET 的横断面、矢状面、冠状面
    参数：
      writer: TensorBoard SummaryWriter
      step: 当前训练 step（或 epoch）
      mri, reconstruction, pet: torch.Tensor，形状为 (C, H, W) 或 (1, C, H, W)，其中 C 通常为 1
    """
    import matplotlib.pyplot as plt

    plt.style.use("dark_background")

    # 如果图像张量多了一个 batch 维度则 squeeze
    if len(mri.shape) == 4:
        mri = mri.squeeze(0)
    if len(reconstruction.shape) == 4:
        reconstruction = reconstruction.squeeze(0)
    if len(pet.shape) == 4:
        pet = pet.squeeze(0)

    # 创建 3 行 3 列的子图，每行显示一种模态的三个视图
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(9, 9))
    for _ax in ax.flatten():
        _ax.set_axis_off()

    mri_views = get_views(mri)
    recon_views = get_views(reconstruction)
    pet_views = get_views(pet)

    # 第一行：MRI
    ax[0, 0].set_title("MRI (Axial)", color="cyan")
    ax[0, 0].imshow(mri_views[0], cmap="gray")
    ax[0, 1].set_title("MRI (Sagittal)", color="cyan")
    ax[0, 1].imshow(mri_views[1], cmap="gray")
    ax[0, 2].set_title("MRI (Coronal)", color="cyan")
    ax[0, 2].imshow(mri_views[2], cmap="gray")

    # 第二行：重建的 PET
    ax[1, 0].set_title("Reconstructed PET (Axial)", color="magenta")
    ax[1, 0].imshow(recon_views[0], cmap="gray")
    ax[1, 1].set_title("Reconstructed PET (Sagittal)", color="magenta")
    ax[1, 1].imshow(recon_views[1], cmap="gray")
    ax[1, 2].set_title("Reconstructed PET (Coronal)", color="magenta")
    ax[1, 2].imshow(recon_views[2], cmap="gray")

    # 第三行：真实 PET
    ax[2, 0].set_title("Real PET (Axial)", color="yellow")
    ax[2, 0].imshow(pet_views[0], cmap="gray")
    ax[2, 1].set_title("Real PET (Sagittal)", color="yellow")
    ax[2, 1].imshow(pet_views[1], cmap="gray")
    ax[2, 2].set_title("Real PET (Coronal)", color="yellow")
    ax[2, 2].imshow(pet_views[2], cmap="gray")

    plt.tight_layout()
    writer.add_figure("CrossModality", fig, global_step=step)


def tb_display_MRI_3PET(writer, step, modality_list):
    """
    在 TensorBoard 中显示跨模态图像对比：
      - 第一行：MRI 的横断面、矢状面、冠状面
      - 第二行：重建的 Aβ-PET 的横断面、矢状面、冠状面
      - 第三行：真实 Aβ-PET 的横断面、矢状面、冠状面
      - 第四行：重建的 Tau-PET 的横断面、矢状面、冠状面
      - 第五行：真实 Tau-PET 的横断面、矢状面、冠状面
      - 第六行：重建的 FDG-PET 的横断面、矢状面、冠状面
      - 第七行：真实 FDG-PET 的横断面、矢状面、冠状面
    参数：
      writer: TensorBoard SummaryWriter
      step: 当前训练 step（或 epoch）
      mri, reconstruction, pet: torch.Tensor，形状为 (C, H, W) 或 (1, C, H, W)，其中 C 通常为 1
    """
    import matplotlib.pyplot as plt

    plt.style.use("dark_background")

    # 创建 3 行 3 列的子图，每行显示一种模态的三个视图
    fig, ax = plt.subplots(nrows=len(modality_list), ncols=3, figsize=(9, 9))
    for _ax in ax.flatten():
        _ax.set_axis_off()

    for i, modality in enumerate(modality_list):
        modality = modality[0].detach().cpu()

        if len(modality.shape) == 4:
            modality = modality.squeeze(0)

        modality_views = get_views(modality)

        ax[i, 0].set_title(f"{modality} (Axial)", color="green")
        ax[i, 0].imshow(modality_views[0], cmap="gray")
        ax[i, 1].set_title(f"{modality} (Sagittal)", color="green")
        ax[i, 1].imshow(modality_views[1], cmap="gray")
        ax[i, 2].set_title(f"{modality} (Coronal)", color="green")
        ax[i, 2].imshow(modality_views[2], cmap="gray")

    plt.tight_layout()
    writer.add_figure("Modalities", fig, global_step=step)


def compute_scale_factors(
    trainset,
    keys=["abeta_pet_latent", "tau_pet_latent", "fdg_pet_latent"],
    sample_size=100,
    device="cuda",
):
    """
    输入:
      - trainset: 训练集数据集（支持索引获取每个样本的字典）
      - keys: 要计算 scale_factor 的键列表（对应不同 PET latent）
      - sample_size: 用于计算统计量的样本数量
    输出:
      - scale_factors: dict, 键为输入keys, 值为计算得到的 scale_factor
    """
    with torch.no_grad():
        with autocast(device_type="cuda", enabled=True):
            latent_values = {key: [] for key in keys}
            count = 0
            for item in trainset:
                for key in keys:
                    latent = item[key].to(device)
                    latent_values[key].append(latent.unsqueeze(0))
                count += 1
                if count >= sample_size:
                    break
    scale_factors = {}
    eps = 1e-8
    for key in keys:
        all_latents = torch.cat(latent_values[key], dim=0)
        std_val = torch.std(all_latents)
        scale_factors[key] = 1.0 / (std_val + eps)
        print(
            f"Key {key}: std={std_val.item():.4f}, scale_factor={scale_factors[key].item():.4f}"
        )
    return scale_factors

