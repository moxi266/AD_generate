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

set_determinism(3407)


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = "cuda:7" if torch.cuda.is_available() else "cpu"
        print(self.device)
        self.start_epoch = 0
        self.total_counter = 0

        # 构造输出和日志目录
        self.output_dir = os.path.join(
            args.output_dir,
            args.pet + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        )

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        self.log_dir = (
            "/staff/limingchao/AD/mycode/u-net/log1/"
            + args.pet
            + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )

        self.writer = SummaryWriter(self.log_dir)

        # 数据预处理和数据加载器
        self.transforms_fn = transforms.Compose(
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

        dataset_df = pd.read_csv(args.dataset_csv)
        train_df = dataset_df[dataset_df.split == "train"]
        valid_df = dataset_df[dataset_df.split == "valid"]

        self.trainset = get_dataset_from_pd(
            train_df, self.transforms_fn, args.cache_dir
        )
        self.validset = get_dataset_from_pd(
            valid_df, self.transforms_fn, args.cache_dir
        )
        self.train_loader = DataLoader(
            dataset=self.trainset,
            num_workers=args.num_workers,
            batch_size=args.max_batch_size,
            shuffle=True,
            persistent_workers=True,
            pin_memory=True,
        )
        self.valid_loader = DataLoader(
            dataset=self.validset,
            num_workers=args.num_workers,
            batch_size=args.max_batch_size,
            shuffle=True,
            persistent_workers=True,
            pin_memory=True,
        )

        # 初始化模型
        self.G = init_Unet(args.G_ckpt).to(self.device)  # MRI → PET

        self.G = self.G.to(self.device)

        # 损失函数
        self.l1_loss_fn = L1Loss()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.perc_loss_fn = PerceptualLoss(
                spatial_dims=3,
                network_type="squeeze",
                is_fake_3d=True,
                fake_3d_ratio=0.2,
            ).to(self.device)

        # 优化器
        self.optimizer_g = torch.optim.Adam(
            self.G.parameters(),
            lr=args.lr,
            betas=(0.5, 0.999),
        )

        # 梯度累积
        self.gradacc_g = GradientAccumulation(
            actual_batch_size=args.max_batch_size,
            expect_batch_size=args.batch_size,
            loader_len=len(self.train_loader),
            optimizer=self.optimizer_g,
            grad_scaler=GradScaler(),
        )

        self.best_valid_mae = float("inf")
        # 均值损失记录器（用于 TensorBoard 日志）
        self.avgloss = utils.AverageLoss()

        # 如果指定了 resume 参数，则加载检查点
        if args.resume is not None:
            self.load_checkpoint(args.resume)

    def save_checkpoint(self, epoch):
        checkpoint = {
            "epoch": epoch,
            "best_valid_mae": self.best_valid_mae,
            "total_counter": self.total_counter,
            "G_state": self.G.state_dict(),
            "optimizer_g_state": self.optimizer_g.state_dict(),
            # 如有需要，也可以保存 gradacc 的状态
        }
        ckpt_path = os.path.join(self.output_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save(checkpoint, ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}")

    def load_checkpoint(self, resume_path):
        if os.path.isfile(resume_path):
            checkpoint = torch.load(resume_path, map_location=self.device)
            # 从检查点中恢复 epoch（下次从 epoch+1 开始训练）
            self.start_epoch = checkpoint.get("epoch", 0) + 1
            self.best_valid_mae = checkpoint.get("best_valid_mae", 0)
            self.total_counter = checkpoint.get("total_counter", 0)
            self.G.load_state_dict(checkpoint["G_state"])
            self.optimizer_g.load_state_dict(checkpoint["optimizer_g_state"])
            print(f"Resumed training from epoch {self.start_epoch}")
        else:
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

    def valid_one_epoch(self, epoch):
        epoch_mae = 0.0
        with torch.no_grad():

            self.G.eval()
            progress_bar = tqdm(
                enumerate(self.valid_loader),
                total=len(self.valid_loader),
                desc="valid:" + f"Epoch: {epoch}",
            )
            for step, batch in progress_bar:
                mri = batch["mri"].to(self.device)
                pet = batch["pet"].to(self.device)
                fake_pet = self.G(mri)
                loss_mae = self.l1_loss_fn(pet.float(), fake_pet.float())
                epoch_mae += loss_mae.item()
            return epoch_mae / len(self.valid_loader)

    def save_model(self, output_dir, epoch):
        torch.save(
            self.G.state_dict(),
            os.path.join(output_dir, f"G-ep-{epoch}.pth"),
        )

    def train(self):
        n_epochs = self.args.n_epochs
        for epoch in range(self.start_epoch, n_epochs):
            torch.cuda.empty_cache()
            self.G.train()
            progress_bar = tqdm(
                enumerate(self.train_loader),
                total=len(self.train_loader),
                desc=f"Epoch {epoch}",
            )
            for step, batch in progress_bar:
                with autocast(device_type=self.device, enabled=True):
                    mri = batch["mri"].to(self.device)
                    pet = batch["pet"].to(self.device)
                    fake_pet = self.G(mri)

                    perceptual_loss = self.perc_loss_fn(
                        fake_pet.float(), pet.float()
                    )

                    l1_loss = self.l1_loss_fn(
                        fake_pet.float(), pet.float()
                    )
                    loss_g =   l1_loss

                self.gradacc_g.step(loss_g, step)

                # 更新日志记录器
                self.avgloss.put("Generator/l1_loss", l1_loss.item())
                self.avgloss.put("Generator/perceptual_loss", perceptual_loss.item())

                if self.total_counter % 10 == 0:
                    current_step = self.total_counter // 10
                    self.avgloss.to_tensorboard(self.writer, current_step)
                    utils.tb_display_MRI_PET(
                        self.writer,
                        current_step,
                        mri[0].detach().cpu(),
                        fake_pet[0].detach().cpu(),
                        pet[0].detach().cpu(),
                    )

                self.total_counter += 1
            valid_mae = self.valid_one_epoch(epoch)
            self.writer.add_scalar("valid/mae", valid_mae, epoch + 1)
            if valid_mae < self.best_valid_mae:
                self.best_valid_mae = valid_mae
                savepath = os.path.join(self.output_dir, "best")
                if not os.path.exists(savepath):
                    os.makedirs(savepath, exist_ok=True)
                self.save_model(savepath, epoch)

            # 定期保存模型与训练检查点
            if epoch % 10 == 0:
                # 单独保存模型
                self.save_model(self.output_dir, epoch)
                self.save_checkpoint(epoch)


def argparse_args():

    parser = argparse.ArgumentParser(description="Train the autoencoder.")
    parser.add_argument(
        "--dataset_csv",
        default="/staff/limingchao/AD/csv/last/abeta-PET_dataset_3.csv",
        type=str,
    )
    parser.add_argument(
        "--cache_dir",
        default="/staff/limingchao/AD/mycode/u-net/cache/",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        default="/staff/limingchao/AD/mycode/u-net/output1/",
        type=str,
    )
    parser.add_argument(
        "--G_ckpt",
        default=None,
        type=str,
    )

    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--n_epochs", default=200, type=int)
    parser.add_argument("--max_batch_size", default=16, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="Path to resume checkpoint",
    )
    parser.add_argument("--pet", default="abeta_", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = argparse_args()
    trainer = Trainer(args)
    trainer.train()
