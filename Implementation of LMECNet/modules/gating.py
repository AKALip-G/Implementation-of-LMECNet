from __future__ import annotations

from typing import Tuple, Union

import torch
import torch.nn as nn


class AdaptiveGate2d(nn.Module):

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=True)

    def forward(
        self,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        return_gate: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if x_a.shape != x_b.shape:
            raise ValueError(f"AdaptiveGate2d shape mismatch: {x_a.shape} vs {x_b.shape}")

        alpha = torch.sigmoid(self.proj(torch.cat([x_a, x_b], dim=1)))
        out = alpha * x_a + (1.0 - alpha) * x_b
        return (out, alpha) if return_gate else out


__all__ = ["AdaptiveGate2d"]