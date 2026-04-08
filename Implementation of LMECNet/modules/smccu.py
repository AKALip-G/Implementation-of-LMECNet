from __future__ import annotations

from typing import Optional, Type

import torch
import torch.nn as nn


from mamba_ssm import Mamba2

from .gating import AdaptiveGate2d
from .sequence_utils import (
    from_col_tokens,
    from_row_tokens,
    to_col_tokens,
    to_row_tokens,
)


class LayerNorm2d(nn.Module):

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


def build_norm_2d(num_channels: int, norm_type: str = "ln2d") -> nn.Module:
    norm_type = norm_type.lower()
    if norm_type == "ln2d":
        return LayerNorm2d(num_channels)
    if norm_type == "bn":
        return nn.BatchNorm2d(num_channels)
    if norm_type == "in":
        return nn.InstanceNorm2d(num_channels, affine=True)
    if norm_type == "identity":
        return nn.Identity()
    raise ValueError(f"Unsupported norm_type={norm_type}")


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Optional[int] = None,
        norm_type: str = "bn",
        act_layer: Type[nn.Module] = nn.GELU,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.norm = build_norm_2d(out_channels, norm_type)
        self.act = act_layer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class Mamba2Sequence(nn.Module):

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        **mamba_kwargs,
    ) -> None:
        super().__init__()
        self.block = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            **mamba_kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Mamba2Sequence expects [B, L, D], got {tuple(x.shape)}")
        return self.block(x)


class SMCCU(nn.Module):
    
    def __init__(
        self,
        dim: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        norm_type: str = "ln2d",
        act_layer: Type[nn.Module] = nn.GELU,
        residual_scale_init: float = 1.0,
        **mamba_kwargs,
    ) -> None:
        super().__init__()
        self.pre_norm = build_norm_2d(dim, norm_type)

        self.local_conv = ConvNormAct(
            dim, dim,
            kernel_size=3,
            stride=1,
            padding=1,
            norm_type="bn",
            act_layer=act_layer,
        )

        self.row_norm = nn.LayerNorm(dim)
        self.col_norm = nn.LayerNorm(dim)

        self.mamba_row = Mamba2Sequence(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            **mamba_kwargs,
        )
        self.mamba_col = Mamba2Sequence(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            **mamba_kwargs,
        )

        self.gate = AdaptiveGate2d(dim)

        self.fuse = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False),
            build_norm_2d(dim, "bn"),
            act_layer(),
        )

        self.gamma = nn.Parameter(torch.full((1, dim, 1, 1), float(residual_scale_init)))

    def _row_branch(self, x2d: torch.Tensor, h: int, w: int) -> torch.Tensor:
        tokens = to_row_tokens(x2d)
        tokens = self.row_norm(tokens)
        tokens = self.mamba_row(tokens)
        return from_row_tokens(tokens, h, w)

    def _col_branch(self, x2d: torch.Tensor, h: int, w: int) -> torch.Tensor:
        tokens = to_col_tokens(x2d)
        tokens = self.col_norm(tokens)
        tokens = self.mamba_col(tokens)
        return from_col_tokens(tokens, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x_norm = self.pre_norm(x)

        local_feat = self.local_conv(x_norm)
        _, _, h, w = local_feat.shape

        row_feat = self._row_branch(local_feat, h, w)
        col_feat = self._col_branch(local_feat, h, w)
        seq_feat = self.gate(row_feat, col_feat)

        out = self.fuse(torch.cat([local_feat, seq_feat], dim=1))
        return shortcut + self.gamma * out


__all__ = [
    "LayerNorm2d",
    "build_norm_2d",
    "ConvNormAct",
    "Mamba2Sequence",
    "SMCCU",
]