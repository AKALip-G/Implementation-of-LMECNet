from __future__ import annotations

import torch


def to_row_tokens(x: torch.Tensor) -> torch.Tensor:
    
    b, c, h, w = x.shape
    return x.reshape(b, c, h * w).transpose(1, 2).contiguous()


def from_row_tokens(tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
    
    b, hw, c = tokens.shape
    if hw != h * w:
        raise ValueError(f"Row token length mismatch: got {hw}, expected {h * w}")
    return tokens.transpose(1, 2).reshape(b, c, h, w).contiguous()


def to_col_tokens(x: torch.Tensor) -> torch.Tensor:
    
    b, c, h, w = x.shape
    x = x.permute(0, 1, 3, 2).contiguous()  # [B, C, W, H]
    return x.reshape(b, c, w * h).transpose(1, 2).contiguous()


def from_col_tokens(tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
    
    b, wh, c = tokens.shape
    if wh != w * h:
        raise ValueError(f"Col token length mismatch: got {wh}, expected {w * h}")
    x = tokens.transpose(1, 2).reshape(b, c, w, h).contiguous()
    x = x.permute(0, 1, 3, 2).contiguous()
    return x


__all__ = [
    "to_row_tokens",
    "from_row_tokens",
    "to_col_tokens",
    "from_col_tokens",
]