from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


TensorDict = Dict[str, torch.Tensor]


class TwinArtifactDiscriminationLoss(nn.Module):


    def __init__(
        self,
        lambda_c: float = 1.0,
        lambda_t: float = 0.5,
        margin: float = 0.15,
    ) -> None:
        super().__init__()
        self.lambda_c = float(lambda_c)
        self.lambda_t = float(lambda_t)
        self.margin = float(margin)

    @staticmethod
    def _check_input(pred: torch.Tensor, target: torch.Tensor) -> None:
        if pred.ndim != 4 or target.ndim != 4:
            raise ValueError("Expected [B,2,H,W] tensors.")
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: {pred.shape} vs {target.shape}")
        if pred.shape[1] != 2:
            raise ValueError(f"Expected 2 channels [real, imag], got {pred.shape[1]}")

    @staticmethod
    def build_twin_hypothesis(target: torch.Tensor) -> torch.Tensor:
        """
        T'(r) = T*(-r):
            real_twin = flip(real)
            imag_twin = -flip(imag)
        """
        real, imag = torch.chunk(target, 2, dim=1)
        real_twin = torch.flip(real, dims=(-2, -1))
        imag_twin = -torch.flip(imag, dims=(-2, -1))
        return torch.cat([real_twin, imag_twin], dim=1)

    @staticmethod
    def complex_l1_per_sample(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        _, _, h, w = pred.shape
        diff = torch.abs(pred - target)
        return diff.sum(dim=(1, 2, 3)) / float(h * w)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> TensorDict:
        self._check_input(pred, target)

        twin = self.build_twin_hypothesis(target)

        d_ref = self.complex_l1_per_sample(pred, target)
        d_twin = self.complex_l1_per_sample(pred, twin)
        repulsion = F.relu(self.margin - d_twin)

        fidelity_term = self.lambda_c * d_ref
        twin_term = self.lambda_t * repulsion
        loss_tad = (fidelity_term + twin_term).mean()

        return {
            "loss_tad": loss_tad,
            "loss_fidelity_log": fidelity_term.mean().detach(),
            "loss_twin_margin_log": twin_term.mean().detach(),
            "distance_ref_log": d_ref.mean().detach(),
            "distance_twin_log": d_twin.mean().detach(),
        }


__all__ = ["TwinArtifactDiscriminationLoss"]