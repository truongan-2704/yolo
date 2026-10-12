"""
C3k2_Titan — Cải tiến khối C3k2 của YOLOv11
============================================
Hợp nhất 4 nghiên cứu khoa học hàng đầu để biến C3k2 thành phiên bản mạnh hơn:

1. FasterNet (CVPR 2023) — Partial Convolution (PConv)
   "Run, Don't Walk: Chasing Higher FLOPS for Faster Neural Networks"
   → PConv chỉ conv trên một phần kênh (1/4), phần còn lại pass-through
     giảm FLOPs/MAC ~75% mà gần như không mất accuracy.

2. StarNet (CVPR 2024) — Star Operation
   "Rewrite the Stars"
   → Phép nhân element-wise (W₁x) ⊙ (W₂x) tương đương ánh xạ vào không
     gian phi tuyến chiều cao implicit (chứng minh bằng khai triển Taylor),
     đem lại sức biểu diễn của MLP wide với chi phí của conv hẹp.

3. RepVGG (CVPR 2021) + RepLKNet (CVPR 2022) — Structural Reparameterization
   → Train với multi-branch (3×3 conv + 7×7 DWConv + identity),
     infer thì merge thành 1 conv duy nhất → giàu kernel khi train,
     không tốn thêm chi phí khi infer.

4. EMA (ICASSP 2023) — Efficient Multi-scale Attention
   "Efficient Multi-Scale Attention Module with Cross-Spatial Learning"
   → Cross-spatial gating cực nhẹ, chỉ cần GroupNorm + GAP.

Block mới giữ NGUYÊN signature của C3k2 để tương thích parser:
    C3k2_Titan(c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True)

Ghi chú: file này độc lập (self-contained), không circular-import với
block.py — TitanBottleneck thay thế Bottleneck/C3k bên trong CSP shell.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════
def _autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class _ConvBNAct(nn.Module):
    """Conv + BN + (SiLU | Identity)."""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, _autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# ════════════════════════════════════════════════════════════════════════
# 1. PartialConv — FasterNet (CVPR 2023)
# ════════════════════════════════════════════════════════════════════════
class PartialConv3x3(nn.Module):
    """
    Partial Convolution: chỉ áp 3×3 conv lên một phần kênh (n_div).
    Phần còn lại bypass nguyên trạng → tiết kiệm cả FLOPs lẫn memory bandwidth.
    """
    def __init__(self, channels, n_div=4):
        super().__init__()
        self.dim_conv = channels // n_div
        self.dim_untouched = channels - self.dim_conv
        self.partial_conv = nn.Conv2d(
            self.dim_conv, self.dim_conv, 3, 1, 1, bias=False
        )

    def forward(self, x):
        x1, x2 = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.partial_conv(x1)
        return torch.cat((x1, x2), dim=1)


# ════════════════════════════════════════════════════════════════════════
# 2. Reparameterizable Multi-Kernel DW Branch — RepVGG / RepLKNet
# ════════════════════════════════════════════════════════════════════════
class RepMultiKernelDW(nn.Module):
    """
    Train: 3 nhánh DW song song (3×3, 7×7, identity) cộng lại.
    Inference: gọi `switch_to_deploy()` để fuse thành 1 DW conv 7×7 duy nhất.
    Lúc chưa fuse, chi phí train cao hơn nhưng AP cao hơn (giàu kernel).
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.dw_small = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.dw_large = nn.Sequential(
            nn.Conv2d(channels, channels, 7, 1, 3, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.identity_bn = nn.BatchNorm2d(channels)
        self.deployed = False
        self.deploy_conv: nn.Conv2d | None = None

    def forward(self, x):
        if self.deployed and self.deploy_conv is not None:
            return self.deploy_conv(x)
        return self.dw_small(x) + self.dw_large(x) + self.identity_bn(x)

    @torch.no_grad()
    def switch_to_deploy(self):
        """Reparameterize: gộp 3 nhánh + BN thành 1 conv 7×7 DW."""
        if self.deployed:
            return
        # Helper: fuse Conv-BN
        def _fuse(conv: nn.Conv2d, bn: nn.BatchNorm2d):
            w = conv.weight
            gamma, beta, mean, var, eps = bn.weight, bn.bias, bn.running_mean, bn.running_var, bn.eps
            std = (var + eps).sqrt()
            t = (gamma / std).reshape(-1, 1, 1, 1)
            return w * t, beta - mean * gamma / std

        # 3×3 → pad lên 7×7
        w3, b3 = _fuse(self.dw_small[0], self.dw_small[1])
        w3_padded = F.pad(w3, [2, 2, 2, 2])

        # 7×7 sẵn sàng
        w7, b7 = _fuse(self.dw_large[0], self.dw_large[1])

        # Identity: tạo kernel one-hot DW 7×7
        id_kernel = torch.zeros(self.channels, 1, 7, 7, device=w7.device, dtype=w7.dtype)
        for i in range(self.channels):
            id_kernel[i, 0, 3, 3] = 1.0
        # Identity BN: chuẩn hoá id_kernel theo running stats của identity_bn
        gamma_i, beta_i = self.identity_bn.weight, self.identity_bn.bias
        mean_i, var_i, eps_i = self.identity_bn.running_mean, self.identity_bn.running_var, self.identity_bn.eps
        std_i = (var_i + eps_i).sqrt()
        t_i = (gamma_i / std_i).reshape(-1, 1, 1, 1)
        w_id = id_kernel * t_i
        b_id = beta_i - mean_i * gamma_i / std_i

        merged_w = w3_padded + w7 + w_id
        merged_b = b3 + b7 + b_id

        self.deploy_conv = nn.Conv2d(
            self.channels, self.channels, 7, 1, 3, groups=self.channels, bias=True
        )
        self.deploy_conv.weight.data = merged_w
        self.deploy_conv.bias.data = merged_b
        self.deployed = True

        # Giải phóng nhánh train
        del self.dw_small, self.dw_large, self.identity_bn


# ════════════════════════════════════════════════════════════════════════
# 3. Star Operation — StarNet (CVPR 2024)
# ════════════════════════════════════════════════════════════════════════
class StarOp(nn.Module):
    """
    Star Operation: y = ReLU(W1·x) ⊙ (W2·x)
    Tương đương ánh xạ vào không gian implicit chiều cao (Ma et al. 2024,
    Theorem 1) mà không cần widen kênh thực.
    """
    def __init__(self, channels, expand=4):
        super().__init__()
        hidden = int(channels * expand)
        self.f1 = _ConvBNAct(channels, hidden, k=1, act=False)
        self.f2 = _ConvBNAct(channels, hidden, k=1, act=False)
        self.act = nn.ReLU6(inplace=True)
        self.proj = _ConvBNAct(hidden, channels, k=1, act=False)

    def forward(self, x):
        a = self.act(self.f1(x))
        b = self.f2(x)
        return self.proj(a * b)


# ════════════════════════════════════════════════════════════════════════
# 4. EMA-Lite gating — EMA (ICASSP 2023), simplified
# ════════════════════════════════════════════════════════════════════════
class EMALite(nn.Module):
    """
    Cross-spatial gating dùng GAP-along-H + GAP-along-W (như CoordAtt) +
    GroupNorm thay BatchNorm để ổn định khi batch nhỏ.
    Cực nhẹ: ~ 2·(C·C/r) params.
    """
    def __init__(self, channels, factor=8):
        super().__init__()
        self.groups = max(1, channels // factor)
        self.gn = nn.GroupNorm(self.groups, channels)
        self.conv1x1 = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        # Pool theo 2 trục
        x_h = F.adaptive_avg_pool2d(x, (h, 1))   # b,c,h,1
        x_w = F.adaptive_avg_pool2d(x, (1, w))   # b,c,1,w
        # Tổng hợp & gate
        gate = torch.sigmoid(self.conv1x1(self.gn(x_h * x_w + x)))
        return x * gate


# ════════════════════════════════════════════════════════════════════════
# 5. TitanBottleneck — thay thế Bottleneck cổ điển
# ════════════════════════════════════════════════════════════════════════
class TitanBottleneck(nn.Module):
    """
    Pipeline:
        x → PConv3×3 → RepMultiKernelDW → StarOp → EMALite → (+x nếu shortcut)

    Ý nghĩa từng bước:
        - PConv:    spatial mixing rẻ tiền (FasterNet).
        - RepDW:    đa kernel khi train, 1 conv khi infer (RepVGG/RepLKNet).
        - StarOp:   ánh xạ implicit chiều cao (StarNet) — phần "channel mixing".
        - EMALite:  gating đa trục, recalibrate kênh + spatial.
    """
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=1.0):
        super().__init__()
        # k và e giữ tương thích chữ ký Bottleneck (không dùng trực tiếp)
        del k, e, g
        self.pconv = PartialConv3x3(c1, n_div=4)
        # Bridge nếu c1 != c2
        self.bridge = _ConvBNAct(c1, c2, k=1) if c1 != c2 else nn.Identity()
        self.repdw = RepMultiKernelDW(c2)
        self.star = StarOp(c2, expand=4)
        self.gate = EMALite(c2)
        self.add = shortcut and (c1 == c2)

    def forward(self, x):
        y = self.pconv(x)
        y = self.bridge(y)
        y = self.repdw(y)
        y = self.star(y)
        y = self.gate(y)
        return x + y if self.add else y


# ════════════════════════════════════════════════════════════════════════
# 6. C3k_Titan — phiên bản C3k dùng TitanBottleneck
# ════════════════════════════════════════════════════════════════════════
class C3k_Titan(nn.Module):
    """CSP với 3 conv + n TitanBottleneck (giống vai trò C3k)."""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = _ConvBNAct(c1, c_, k=1)
        self.cv2 = _ConvBNAct(c1, c_, k=1)
        self.cv3 = _ConvBNAct(2 * c_, c2, k=1)
        self.m = nn.Sequential(*(TitanBottleneck(c_, c_, shortcut=shortcut) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


# ════════════════════════════════════════════════════════════════════════
# 7. C3k2_Titan — drop-in replacement của C3k2
# ════════════════════════════════════════════════════════════════════════
class C3k2_Titan(nn.Module):
    """
    Drop-in cho C3k2 (giữ NGUYÊN signature):
        C3k2_Titan(c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True)

    Cấu trúc giống C2f gốc:
        cv1 1×1 → split 2 → n × Titan blocks → concat → cv2 1×1
    Khi c3k=True, thay TitanBottleneck bằng C3k_Titan (CSP nhỏ trong CSP lớn).
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = _ConvBNAct(c1, 2 * self.c, k=1)
        self.cv2 = _ConvBNAct((2 + n) * self.c, c2, k=1)
        if c3k:
            self.m = nn.ModuleList(C3k_Titan(self.c, self.c, n=2, shortcut=shortcut) for _ in range(n))
        else:
            self.m = nn.ModuleList(TitanBottleneck(self.c, self.c, shortcut=shortcut) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def switch_to_deploy(self):
        """Gọi recursive để fuse RepMultiKernelDW thành conv 7×7 deploy mode."""
        for m in self.modules():
            if isinstance(m, RepMultiKernelDW):
                m.switch_to_deploy()


__all__ = [
    "PartialConv3x3",
    "RepMultiKernelDW",
    "StarOp",
    "EMALite",
    "TitanBottleneck",
    "C3k_Titan",
    "C3k2_Titan",
]
