from __future__ import annotations

from typing import Sequence, Tuple

import torch


def fft2c(x: torch.Tensor) -> torch.Tensor:
    
    x = torch.fft.fft2(x, norm="ortho")
    x = torch.fft.fftshift(x, dim=(-2, -1))
    return x


def ifft2c(x: torch.Tensor) -> torch.Tensor:
    
    x = torch.fft.ifftshift(x, dim=(-2, -1))
    x = torch.fft.ifft2(x, norm="ortho")
    return x


def complex_to_channels(x: torch.Tensor) -> torch.Tensor:
    
    if not torch.is_complex(x):
        raise TypeError("complex_to_channels expects a complex tensor.")
    return torch.cat([x.real, x.imag], dim=1)


def channels_to_complex(x: torch.Tensor) -> torch.Tensor:
    
    if x.ndim != 4 or x.shape[1] % 2 != 0:
        raise ValueError(f"channels_to_complex expects [B,2C,H,W], got {tuple(x.shape)}")
    real, imag = torch.chunk(x, 2, dim=1)
    return torch.complex(real, imag)


def build_radial_grid(
    h: int,
    w: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    
    yy = torch.linspace(-1.0, 1.0, steps=h, device=device, dtype=dtype)
    xx = torch.linspace(-1.0, 1.0, steps=w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(yy, xx, indexing="ij")
    rr = torch.sqrt(xx.square() + yy.square())
    rr = rr / rr.max().clamp_min(1e-12)
    return rr.unsqueeze(0).unsqueeze(0)


def build_band_masks(
    h: int,
    w: int,
    bands: Sequence[Tuple[float, float]],
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, ...]:
    rr = build_radial_grid(h, w, device=device, dtype=dtype)
    masks = []
    for idx, (r0, r1) in enumerate(bands):
        if idx < len(bands) - 1:
            mask = (rr >= r0) & (rr < r1)
        else:
            mask = (rr >= r0) & (rr <= r1)
        masks.append(mask.to(dtype))
    return tuple(masks)


def reassemble_bands(
    band_outputs: Sequence[torch.Tensor],
    band_masks: Sequence[torch.Tensor],
) -> torch.Tensor:
    
    if len(band_outputs) == 0:
        raise ValueError("band_outputs must be non-empty")
    if len(band_outputs) != len(band_masks):
        raise ValueError("band_outputs and band_masks must have the same length")

    out = torch.zeros_like(band_outputs[0])
    for y, m in zip(band_outputs, band_masks):
        out = out + y * m
    return out


__all__ = [
    "fft2c",
    "ifft2c",
    "complex_to_channels",
    "channels_to_complex",
    "build_radial_grid",
    "build_band_masks",
    "reassemble_bands",
]