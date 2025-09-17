import os
import sys
from pathlib import Path
from typing import Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

QUATNET_PATH = os.environ.get('QUATNET_PATH')
if QUATNET_PATH:
    quatnet_path = Path(QUATNET_PATH).expanduser().resolve()
    if str(quatnet_path) not in sys.path:
        sys.path.insert(0, str(quatnet_path))

try:
    from quaternion_layers import QuaternionConv, QuaternionTransposeConv
except ImportError as exc:
    raise ImportError(
        'quaternion_layers.py is required. Set the QUATNET_PATH environment '
        'variable to the directory containing the Quanterion quaternion layers.'
    ) from exc

# Helper modules from the original notebook implementation.
class _QConvNormLReLU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[Sequence[int], int], stride: Union[Sequence[int], int], norm_layer: nn.Module = nn.BatchNorm3d, act_layer: nn.Module = nn.LeakyReLU, padding: Union[Sequence[int], int, str] = "same", bias: bool = False, init_criterion: str = 'he', weight_init: str = 'quaternion'):
        q_in_real, q_out_real = in_channels * 4, out_channels * 4
        if padding == "same":
            kernel_size_tuple = (kernel_size,) * 3 if isinstance(kernel_size, int) else kernel_size
            padding_val = tuple((k - 1) // 2 for k in kernel_size_tuple)
        else: padding_val = padding
        q_conv = QuaternionConv(in_channels=q_in_real, out_channels=q_out_real, kernel_size=kernel_size, stride=stride, padding=padding_val, bias=bias, operation='convolution3d', init_criterion=init_criterion, weight_init=weight_init)
        norm = norm_layer(q_out_real) if norm_layer else nn.Identity()
        act = act_layer() if act_layer else nn.Identity()
        super().__init__(q_conv, norm, act)

class _UpBlockTranspose(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[Sequence[int], int], stride: Union[Sequence[int], int], norm_layer: nn.Module = nn.BatchNorm3d, act_layer: nn.Module = nn.LeakyReLU, output_padding: Union[Sequence[int], int] = 0, bias: bool = False, init_criterion: str = 'he', weight_init: str = 'quaternion'):
        q_in_real, q_out_real = in_channels * 4, out_channels * 4
        kernel_size_tup = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 3
        stride_tup = stride if isinstance(stride, tuple) else (stride,) * 3
        padding_val = tuple((k - s) // 2 for k, s in zip(kernel_size_tup, stride_tup))
        q_tconv = QuaternionTransposeConv(in_channels=q_in_real, out_channels=q_out_real, kernel_size=kernel_size, stride=stride, padding=padding_val, output_padding=output_padding, bias=bias, operation='convolution3d', init_criterion=init_criterion, weight_init=weight_init)
        norm = norm_layer(q_out_real) if norm_layer else nn.Identity()
        act = act_layer() if act_layer else nn.Identity()
        super().__init__(q_tconv, norm, act)

class QuaternionAttentionGate(nn.Module):
    def __init__(self, g_channels: int, x_channels: int, inter_channels: int):
        super().__init__(); self.g_channels_real, self.x_channels_real, self.inter_channels_real = g_channels * 4, x_channels * 4, inter_channels * 4
        self.W_g = QuaternionConv(self.g_channels_real, self.inter_channels_real, kernel_size=1, stride=1, padding=0, bias=False, operation='convolution3d')
        self.bn_g = nn.BatchNorm3d(self.inter_channels_real)
        self.W_x = QuaternionConv(self.x_channels_real, self.inter_channels_real, kernel_size=1, stride=1, padding=0, bias=False, operation='convolution3d')
        self.bn_x = nn.BatchNorm3d(self.inter_channels_real)
        self.psi = nn.Sequential(nn.LeakyReLU(), nn.Conv3d(self.inter_channels_real, 1, kernel_size=1, stride=1, padding=0, bias=True), nn.BatchNorm3d(1), nn.Sigmoid())
        self.relu = nn.LeakyReLU()
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g_proj, x_proj = self.bn_g(self.W_g(g)), self.bn_x(self.W_x(x))
        combined = self.relu(g_proj + x_proj)
        alpha = self.psi(combined)
        return alpha * x
    
class AdaptiveGatingUnit(nn.Module):
    def __init__(self, quat_channels: int, reduction_ratio: int = 4):
        super().__init__()
        real_channels = quat_channels * 4
        reduced_channels = real_channels // reduction_ratio

        self.gap = nn.AdaptiveAvgPool3d(1)
        self.mlp = nn.Sequential(
            nn.Conv3d(real_channels * 2, reduced_channels, kernel_size=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(reduced_channels, real_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x_original: torch.Tensor, x_context: torch.Tensor) -> torch.Tensor:
        s_original = self.gap(x_original)
        s_context = self.gap(x_context)
        
        s_combined = torch.cat([s_original, s_context], dim=1)
        gate = self.mlp(s_combined)
        
        return gate
    
class AdaptiveQCFBlock(nn.Module):
    def __init__(self, quat_channels: int):
        super().__init__()
        real_channels = quat_channels * 4
        
        self.q_proj = QuaternionConv(real_channels, real_channels, kernel_size=1, stride=1, operation='convolution3d')
        self.k_proj = QuaternionConv(real_channels, real_channels, kernel_size=1, stride=1, operation='convolution3d')
        self.v_proj = QuaternionConv(real_channels, real_channels, kernel_size=1, stride=1, operation='convolution3d')

        self.gate_mri_to_ct = AdaptiveGatingUnit(quat_channels)
        self.gate_ct_to_mri = AdaptiveGatingUnit(quat_channels)
        
  
        self.output_proj_ct = _QConvNormLReLU(quat_channels * 2, quat_channels, kernel_size=3, stride=1)
        self.output_proj_mri = _QConvNormLReLU(quat_channels * 2, quat_channels, kernel_size=3, stride=1)

    def forward(self, x_ct: torch.Tensor, x_mri: torch.Tensor) -> (torch.Tensor, torch.Tensor):

        q_ct, k_mri, v_mri = self.q_proj(x_ct), self.k_proj(x_mri), self.v_proj(x_mri)
        mri_context_raw = F.softmax(q_ct * k_mri, dim=1) * v_mri
        
        gate_val_mri_ct = self.gate_mri_to_ct(x_original=x_ct, x_context=mri_context_raw)
        mri_context_gated = mri_context_raw * gate_val_mri_ct

        updated_ct = self.output_proj_ct(torch.cat([x_ct, mri_context_gated], dim=1))

        q_mri, k_ct, v_ct = self.q_proj(x_mri), self.k_proj(x_ct), self.v_proj(x_ct)
        ct_context_raw = F.softmax(q_mri * k_ct, dim=1) * v_ct

        gate_val_ct_mri = self.gate_ct_to_mri(x_original=x_mri, x_context=ct_context_raw)
        ct_context_gated = ct_context_raw * gate_val_ct_mri

        updated_mri = self.output_proj_mri(torch.cat([x_mri, ct_context_gated], dim=1))
        
        return updated_ct, updated_mri
    
class AQCF_Net(nn.Module):
    def __init__(self, spatial_dims: int = 3, in_channels: int = 1, out_channels: int = 3, channels: Sequence[int] = (16, 32, 64, 128, 256), strides: Sequence[int] = (2, 2, 2, 2), kernel_size: int = 3, up_kernel_size: int = 2):
        super().__init__()
        norm_layer, act_layer = nn.BatchNorm3d, nn.LeakyReLU
        self.ct_input_proj = nn.Sequential(nn.Conv3d(in_channels, channels[0] * 4, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2, bias=False), norm_layer(channels[0] * 4), act_layer())
        self.mri_input_proj = nn.Sequential(nn.Conv3d(in_channels, channels[0] * 4, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2, bias=False), norm_layer(channels[0] * 4), act_layer())
        self.ct_down_blocks, self.mri_down_blocks, self.fusion_blocks = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for i in range(len(channels) - 1):
            in_ch, out_ch = channels[i], channels[i+1]
            self.ct_down_blocks.append(nn.Sequential(_QConvNormLReLU(in_ch, out_ch, kernel_size=kernel_size, stride=strides[i]), _QConvNormLReLU(out_ch, out_ch, kernel_size=kernel_size, stride=1)))
            self.mri_down_blocks.append(nn.Sequential(_QConvNormLReLU(in_ch, out_ch, kernel_size=kernel_size, stride=strides[i]), _QConvNormLReLU(out_ch, out_ch, kernel_size=kernel_size, stride=1)))
            self.fusion_blocks.append(QuaternionCrossFusionBlock(quat_channels=out_ch))
        self.bottleneck = nn.Sequential(_QConvNormLReLU(channels[-1], channels[-1], kernel_size=kernel_size, stride=1))
        self.ct_up_blocks, self.mri_up_blocks, self.ct_attention_gates, self.mri_attention_gates = nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for i in reversed(range(len(channels) - 1)):
            in_ch, out_ch = channels[i+1], channels[i]
            self.ct_attention_gates.append(QuaternionAttentionGate(g_channels=out_ch, x_channels=out_ch, inter_channels=out_ch // 2))
            self.ct_up_blocks.append(_UpBlockTranspose(in_ch, out_ch, kernel_size=up_kernel_size, stride=strides[i]))
            self.ct_up_blocks.append(nn.Sequential(_QConvNormLReLU(out_ch * 2, out_ch, kernel_size=kernel_size, stride=1), _QConvNormLReLU(out_ch, out_ch, kernel_size=kernel_size, stride=1)))
            self.mri_attention_gates.append(QuaternionAttentionGate(g_channels=out_ch, x_channels=out_ch, inter_channels=out_ch // 2))
            self.mri_up_blocks.append(_UpBlockTranspose(in_ch, out_ch, kernel_size=up_kernel_size, stride=strides[i]))
            self.mri_up_blocks.append(nn.Sequential(_QConvNormLReLU(out_ch * 2, out_ch, kernel_size=kernel_size, stride=1), _QConvNormLReLU(out_ch, out_ch, kernel_size=kernel_size, stride=1)))
        self.ct_output_conv = nn.Conv3d(channels[0] * 4, out_channels, kernel_size=1)
        self.mri_output_conv = nn.Conv3d(channels[0] * 4, out_channels, kernel_size=1)

    def forward(self, x_in=None, ct_images=None, mri_images=None, modality=None):
        if self.training:
            if ct_images is None or mri_images is None:
                raise ValueError("Both ct_images and mri_images are required for training.")
            
            x_ct = self.ct_input_proj(ct_images)
            x_mri = self.mri_input_proj(mri_images)

        else:
            if modality is None:
                raise ValueError("The 'modality' flag ('CT' or 'MRI') is required for inference.")
            if x_in is None:
                if modality.upper() == "CT": x_in = ct_images
                elif modality.upper() == "MRI": x_in = mri_images
            
            if x_in is None:
                 raise ValueError("A single input tensor is required for inference (e.g., x_in=..., modality=...).")

            if modality.upper() == "CT":
                ct_images = x_in
                mri_images = torch.zeros_like(ct_images)
            elif modality.upper() == "MRI":
                mri_images = x_in
                ct_images = torch.zeros_like(mri_images)
            else:
                raise ValueError(f"Unknown modality for inference: {modality}")
                
            x_ct = self.ct_input_proj(ct_images)
            x_mri = self.mri_input_proj(mri_images)
        skips_ct, skips_mri = [], []
        skips_ct.append(x_ct); skips_mri.append(x_mri)

        for i in range(len(self.ct_down_blocks)):
            x_ct, x_mri = self.ct_down_blocks[i](x_ct), self.mri_down_blocks[i](x_mri)
            x_ct, x_mri = self.fusion_blocks[i](x_ct, x_mri)
            skips_ct.append(x_ct); skips_mri.append(x_mri)
        x_ct, x_mri = self.bottleneck(x_ct), self.bottleneck(x_mri)
        skips_ct, skips_mri = skips_ct[::-1], skips_mri[::-1]

        for i in range(len(self.ct_down_blocks)):
            up_conv_ct, conv_block_ct, att_gate_ct = self.ct_up_blocks[i * 2], self.ct_up_blocks[i * 2 + 1], self.ct_attention_gates[i]
            gating_signal_ct = up_conv_ct(x_ct)
            if gating_signal_ct.shape[2:] != skips_ct[i+1].shape[2:]: gating_signal_ct = F.interpolate(gating_signal_ct, size=skips_ct[i+1].shape[2:], mode='trilinear', align_corners=False)
            attended_skip_ct = att_gate_ct(g=gating_signal_ct, x=skips_ct[i+1])
            x_ct = conv_block_ct(torch.cat((attended_skip_ct, gating_signal_ct), dim=1))

            up_conv_mri, conv_block_mri, att_gate_mri = self.mri_up_blocks[i * 2], self.mri_up_blocks[i * 2 + 1], self.mri_attention_gates[i]
            gating_signal_mri = up_conv_mri(x_mri)
            if gating_signal_mri.shape[2:] != skips_mri[i+1].shape[2:]: gating_signal_mri = F.interpolate(gating_signal_mri, size=skips_mri[i+1].shape[2:], mode='trilinear', align_corners=False)
            attended_skip_mri = att_gate_mri(g=gating_signal_mri, x=skips_mri[i+1])
            x_mri = conv_block_mri(torch.cat((attended_skip_mri, gating_signal_mri), dim=1))

        if self.training:
            return self.ct_output_conv(x_ct), self.mri_output_conv(x_mri)
        else:
            if modality.upper() == "CT":
                return self.ct_output_conv(x_ct)
            else:
                return self.mri_output_conv(x_mri)


def create_aqcfnet(input_channels=1, output_classes=3, quat_channels=(12, 24, 48, 96, 192), strides=(2, 2, 2, 2)):
    return AQCF_Net(
        input_channels=input_channels,
        output_classes=output_classes,
        quat_channels=quat_channels,
        strides=strides
    )
