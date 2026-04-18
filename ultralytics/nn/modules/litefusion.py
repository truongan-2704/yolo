# LiteFusion: weighted multi-input fusion + depthwise spatial mix + ECA gating + residual
# Dùng trong neck YOLO11 hybrid. Tương thích parse_model của Ultralytics: c1 là list channels.
"""
LiteFusion module.

Gộp 3 việc thường tách rời ở neck YOLO (channel-align 1x1, weighted fusion kiểu BiFPN,
channel attention ECA) thành một nn.Module duy nhất, cộng thêm residual từ nhánh chính.

YAML usage:
    - [[from_a, from_b, ...], 1, LiteFusion, [c_out]]

parse_model tự inject c1 = [ch[x] for x in f] và c2 = make_divisible(c_out*width, 8).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv

__all__ = ["LiteFusion"]


class LiteFusion(nn.Module):
    """Lightweight weighted fusion block for YOLO neck.

    Args:
        c1 (list[int]): Channel số của từng input tensor.
        c2 (int): Channel output.
        eca_k (int, optional): Kernel size cho ECA 1D conv. Auto nếu None.
    """

    def __init__(self, c1, c2, eca_k=None):
        super().__init__()
        assert isinstance(c1, (list, tuple)) and len(c1) >= 1, \
            f"LiteFusion expects c1 as list of input channels, got {c1}"
        self.n = len(c1)
        self.c2 = c2

        # 1x1 project mỗi input về c2
        self.proj = nn.ModuleList([Conv(c, c2, k=1, s=1) for c in c1])

        # Fast-normalized weights (BiFPN-style)
        self.w = nn.Parameter(torch.ones(self.n, dtype=torch.float32), requires_grad=True)
        self.eps = 1e-4

        # Depthwise 3x3 spatial mixing + pointwise refine
        self.dw = Conv(c2, c2, k=3, s=1, g=c2)   # depthwise
        self.pw = Conv(c2, c2, k=1, s=1)          # pointwise

        # ECA: 1D conv trên channel axis sau GAP
        if eca_k is None:
            # Heuristic kernel (paper ECA): t = |log2(C)/gamma + b/gamma|, odd
            t = int(abs(math.log2(max(c2, 2)) / 2 + 1))
            eca_k = t if t % 2 else t + 1
        self.eca = nn.Conv1d(1, 1, kernel_size=eca_k, padding=eca_k // 2, bias=False)

    def forward(self, x):
        # x là list tensors
        if not isinstance(x, (list, tuple)):
            x = [x]
        assert len(x) == self.n, f"LiteFusion: expected {self.n} inputs, got {len(x)}"

        # Align spatial về shape của input đầu tiên
        h0, w0 = x[0].shape[-2:]
        feats = []
        for i, xi in enumerate(x):
            if xi.shape[-2] != h0 or xi.shape[-1] != w0:
                xi = F.interpolate(xi, size=(h0, w0), mode="nearest")
            feats.append(self.proj[i](xi))

        # Normalized weights
        w = F.relu(self.w)
        w = w / (w.sum() + self.eps)

        # Weighted sum
        y = feats[0] * w[0]
        for i in range(1, self.n):
            y = y + feats[i] * w[i]

        # Depthwise spatial mix + pointwise
        y = self.pw(self.dw(y))

        # ECA channel gate
        # GAP -> (B, C, 1, 1) -> (B, 1, C) cho Conv1d
        g = F.adaptive_avg_pool2d(y, 1).squeeze(-1).transpose(-1, -2)   # (B, 1, C)
        g = torch.sigmoid(self.eca(g)).transpose(-1, -2).unsqueeze(-1)  # (B, C, 1, 1)
        y = y * g

        # Residual với nhánh chính (feats[0]) — đã cùng c2, cùng HxW
        y = y + feats[0]
        return y
