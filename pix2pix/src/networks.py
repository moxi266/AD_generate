import os
from typing import Optional

import torch
import torch.nn as nn
from generative.networks.nets import (
    AutoencoderKL,
    PatchDiscriminator,
    DiffusionModelUNet,
    ControlNet,
)
from monai.networks.nets import AttentionUnet


def load_if(checkpoints_path: Optional[str], network: nn.Module) -> nn.Module:
    """
    Load pretrained weights if available.

    Args:
        checkpoints_path (Optional[str]): path of the checkpoints
        network (nn.Module): the neural network to initialize

    Returns:
        nn.Module: the initialized neural network
    """
    if checkpoints_path is not None:
        assert os.path.exists(checkpoints_path), "Invalid path"
        print(f"Loading pretrained weights from {checkpoints_path}")
        device = next(network.parameters()).device  # Use the same device as the model
        network.load_state_dict(torch.load(checkpoints_path, map_location=device))

    return network


def init_autoencoder(checkpoints_path: Optional[str] = None) -> nn.Module:
    """
    Load the KL autoencoder (pretrained if `checkpoints_path` points to previous params).

    Args:
        checkpoints_path (Optional[str], optional): path of the checkpoints. Defaults to None.

    Returns:
        nn.Module: the KL autoencoder
    """
    autoencoder = AutoencoderKL(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        latent_channels=3,
        num_channels=(64, 128, 128, 128),
        num_res_blocks=2,
        norm_num_groups=32,
        norm_eps=1e-06,
        attention_levels=(False, False, False, False),
        with_decoder_nonlocal_attn=False,
        with_encoder_nonlocal_attn=False,
    )
    return load_if(checkpoints_path, autoencoder)


def init_patch_discriminator(checkpoints_path: Optional[str] = None) -> nn.Module:
    """
    Load the patch discriminator (pretrained if `checkpoints_path` points to previous params).

    Args:
        checkpoints_path (Optional[str], optional): path of the checkpoints. Defaults to None.

    Returns:
        nn.Module: the parch discriminator
    """
    patch_discriminator = PatchDiscriminator(
        spatial_dims=3, num_layers_d=3, num_channels=32, in_channels=2, out_channels=1
    )
    return load_if(checkpoints_path, patch_discriminator)


def init_latent_diffusion(checkpoints_path: Optional[str] = None) -> nn.Module:
    """
    Load the UNet from the diffusion model (pretrained if `checkpoints_path` points to previous params).

    Args:
        checkpoints_path (Optional[str], optional): path of the checkpoints. Defaults to None.

    Returns:
        nn.Module: the UNet
    """
    latent_diffusion = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=3,
        out_channels=3,
        num_res_blocks=2,
        num_channels=(256, 512, 768),
        attention_levels=(False, True, True),
        norm_num_groups=32,
        norm_eps=1e-6,
        resblock_updown=True,
        num_head_channels=(0, 512, 768),
        transformer_num_layers=1,
        with_conditioning=True,
        cross_attention_dim=3,
        num_class_embeds=None,
        upcast_attention=True,
        use_flash_attention=False,
    )
    return load_if(checkpoints_path, latent_diffusion)


def init_controlnet(checkpoints_path: Optional[str] = None) -> nn.Module:
    """
    Load the ControlNet (pretrained if `checkpoints_path` points to previous params).

    Args:
        checkpoints_path (Optional[str], optional): path of the checkpoints. Defaults to None.

    Returns:
        nn.Module: the ControlNet
    """
    controlnet = ControlNet(
        spatial_dims=3,
        in_channels=3,
        num_res_blocks=2,
        num_channels=(256, 512, 768),
        attention_levels=(False, True, True),
        norm_num_groups=32,
        norm_eps=1e-6,
        resblock_updown=True,
        num_head_channels=(0, 512, 768),
        transformer_num_layers=1,
        with_conditioning=True,
        cross_attention_dim=8,
        num_class_embeds=None,
        upcast_attention=True,
        use_flash_attention=False,
        conditioning_embedding_in_channels=4,
        conditioning_embedding_num_channels=(256,),
    )
    return load_if(checkpoints_path, controlnet)


import torch.nn as nn


class UNet3D_Generator(nn.Module):
    def __init__(self):
        super(UNet3D_Generator, self).__init__()

        def conv_block(in_c, out_c, kernel=3, stride=1, padding=1):
            return nn.Sequential(
                nn.Conv3d(
                    in_c, out_c, kernel_size=kernel, stride=stride, padding=padding
                ),
                nn.BatchNorm3d(out_c),
                nn.ReLU(inplace=True),
            )

        def up_block(in_c, out_c):
            return nn.Sequential(
                nn.ConvTranspose3d(in_c, out_c, kernel_size=2, stride=2),
                nn.BatchNorm3d(out_c),
                nn.ReLU(inplace=True),
            )

        self.encoder = nn.Sequential(
            conv_block(1, 32),
            conv_block(32, 64),
            nn.MaxPool3d(2),
            conv_block(64, 128),
            nn.MaxPool3d(2),
            conv_block(128, 256),
        )

        self.decoder = nn.Sequential(
            up_block(256, 128),
            conv_block(128, 64),
            up_block(64, 32),
            conv_block(32, 16),
            nn.Conv3d(16, 1, kernel_size=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def init_AttentionUnet(checkpoints_path: Optional[str] = None) -> nn.Module:
    """
    Load the AttentionUnet (pretrained if `checkpoints_path` points to previous params).

    Args:
        checkpoints_path (Optional[str], optional): path of the checkpoints. Defaults to None.

    Returns:
        nn.Module: the AttentionUnet
    """
    attentionUnet = AttentionUnet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=[32, 64, 128, 256],
        strides=[2, 2, 2, 2],
        kernel_size=3,
        up_kernel_size=3,
        dropout=0.1,
    )
    return load_if(checkpoints_path, attentionUnet)
