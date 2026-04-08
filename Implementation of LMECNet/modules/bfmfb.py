from __future__ import annotations

from typing import Sequence, Tuple, Type

import torch
import torch.nn as nn

from .gating import AdaptiveGate2d
from .sequence_utils import (
    from_col_tokens,
    from_row_tokens,
    to_col_tokens,
    to_row_tokens,
)
from .smccu import Mamba2Sequence, build_norm_2d
from .spectral_utils import (
    build_band_masks,
    channels_to_complex,
    complex_to_channels,
    fft2c,
    ifft2c,
    reassemble_bands,
)


class BFMFB(nn.Module):
    
    def __init__(
        self,
        dim: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        bands: Sequence[Tuple[float, float]] = ((0.0, 0.25), (0.25, 0.5), (0.5, 1.0)),
        norm_type: str = "ln2d",
        act_layer: Type[nn.Module] = nn.GELU,
        residual_scale_init: float = 1.0,
        **mamba_kwargs,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.pair_dim = dim * 2
        self.bands = tuple(bands)

        self.pre_norm = build_norm_2d(dim, norm_type)

        self.band_pre = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.pair_dim, self.pair_dim, kernel_size=1, bias=False),
                build_norm_2d(self.pair_dim, "bn"),
                act_layer(),
            )
            for _ in self.bands
        ])

        self.band_post = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.pair_dim, self.pair_dim, kernel_size=1, bias=False),
                build_norm_2d(self.pair_dim, "bn"),
            )
            for _ in self.bands
        ])

        self.row_norms = nn.ModuleList([nn.LayerNorm(self.pair_dim) for _ in self.bands])
        self.col_norms = nn.ModuleList([nn.LayerNorm(self.pair_dim) for _ in self.bands])

        self.shared_row_mamba = Mamba2Sequence(
            d_model=self.pair_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            **mamba_kwargs,
        )
        self.shared_col_mamba = Mamba2Sequence(
            d_model=self.pair_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            **mamba_kwargs,
        )

        self.band_gates = nn.ModuleList([
            AdaptiveGate2d(self.pair_dim) for _ in self.bands
        ])

        self.spatial_proj = nn.Sequential(
            nn.Conv2d(self.pair_dim, dim, kernel_size=1, bias=False),
            build_norm_2d(dim, "bn"),
            act_layer(),
        )

        self.gamma = nn.Parameter(torch.full((1, dim, 1, 1), float(residual_scale_init)))

    def _row_branch(self, x2d: torch.Tensor, h: int, w: int, idx: int) -> torch.Tensor:
        tokens = to_row_tokens(x2d)
        tokens = self.row_norms[idx](tokens)
        tokens = self.shared_row_mamba(tokens)
        return from_row_tokens(tokens, h, w)

    def _col_branch(self, x2d: torch.Tensor, h: int, w: int, idx: int) -> torch.Tensor:
        tokens = to_col_tokens(x2d)
        tokens = self.col_norms[idx](tokens)
        tokens = self.shared_col_mamba(tokens)
        return from_col_tokens(tokens, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.pre_norm(x)

        _, _, h, w = x.shape
        x_fft = fft2c(x)
        x_pair = complex_to_channels(x_fft)

        masks = build_band_masks(
            h=h,
            w=w,
            bands=self.bands,
            device=x_pair.device,
            dtype=x_pair.dtype,
        )

        enhanced_bands = []
        for idx, mask in enumerate(masks):
            x_band = x_pair * mask
            x_band = self.band_pre[idx](x_band)

            y_row = self._row_branch(x_band, h, w, idx)
            y_col = self._col_branch(x_band, h, w, idx)

            y_band = self.band_gates[idx](y_row, y_col)
            y_band = self.band_post[idx](y_band)
            enhanced_bands.append(y_band)

        x_pair_out = reassemble_bands(enhanced_bands, masks)
        x_fft_out = channels_to_complex(x_pair_out)

        x_spatial_complex = ifft2c(x_fft_out)
        x_spatial_pair = complex_to_channels(x_spatial_complex)
        out = self.spatial_proj(x_spatial_pair)

        return shortcut + self.gamma * out


__all__ = ["BFMFB"]