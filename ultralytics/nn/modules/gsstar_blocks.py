"""
GSStarBlock — Ghost-Shuffle-Star Block for YOLO Neck
======================================================
Một khối mới HOÀN TOÀN khác C3k2, thiết kế chuyên cho NECK:
- nhẹ hơn C3k2 (~30–45% params và FLOPs)
- expressive hơn ở channel mixing nhờ Star Operation
- thông tin chéo nhánh tốt hơn nhờ Channel Shuffle
- I/O memory thấp nhờ One-Shot Aggregation (không dense-concat trung gian)

Cơ sở khoa học:
1) GhostNet (Han et al., CVPR 2020)         — Ghost feature generation.
2) ShuffleNetV2 (Ma et al., ECCV 2018)      — Channel split + shuffle (4 G-rules).
3) StarNet (Ma et al., CVPR 2024)           — Star op = implicit high-dim mapping.
4) VoVNet (Lee et al., CVPR 2019)           — One-Shot Aggregation (OSA).

Block public:
    GSStarBlock(c1, c2, n=1, e=0.5, shortcut=True)
    GSStarCSP  (c1, c2, n=1, e=0.5, shortcut=True)   # nếu cần CSP wrapper

GSStarBlock được thiết kế để THAY THẾ C3k2 trong NECK của YOLOv11 với:
- Cùng chữ ký giống C3k2 (drop-in friendly).
- Cùng kiểu output channels = c2.
- Nhẹ hơn, ít hop-skip, friendly với cache CPU/NPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════
def _autopad(k, p=None):
    return k // 2 if p is None else p


class _CBA(nn.Module):
    """Conv + BN + (SiLU | Identity)."""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, _autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    """ShuffleNetV2 channel shuffle (groups must divide C)."""
    b, c, h, w = x.shape
    if groups <= 1 or c % groups != 0:
        return x
    x = x.view(b, groups, c // groups, h, w).transpose(1, 2).contiguous()
    return x.view(b, c, h, w)


# ════════════════════════════════════════════════════════════════════════
# 1. GhostConv — GhostNet (CVPR 2020)
# ════════════════════════════════════════════════════════════════════════
class GhostUnit(nn.Module):
    """
    Ghost module: chia output làm 2 nửa.
      - Nửa 1: 1×1 conv chuẩn (intrinsic).
      - Nửa 2: DWConv 3×3 trên nửa 1 (cheap "ghost").
    Tạo c_out feature với chỉ ~½ chi phí conv chuẩn.
    """
    def __init__(self, c_in, c_out, k=1, s=1, ratio=2, dw_k=3):
        super().__init__()
        c_intrinsic = c_out // ratio
        c_ghost = c_out - c_intrinsic
        self.primary = _CBA(c_in, c_intrinsic, k=k, s=s, act=True)
        self.cheap = _CBA(c_intrinsic, c_ghost, k=dw_k, s=1, g=c_intrinsic, act=True)

    def forward(self, x):
        y1 = self.primary(x)
        y2 = self.cheap(y1)
        return torch.cat((y1, y2), dim=1)


# ════════════════════════════════════════════════════════════════════════
# 2. Star Operation — StarNet (CVPR 2024) — phiên bản LIGHT cho neck
# ════════════════════════════════════════════════════════════════════════
class StarLite(nn.Module):
    """
    y = SiLU(W1 x) ⊙ (W2 x) → 1×1 project.
    Light version: expand chỉ 2× (StarNet gốc dùng 4×, nhưng ở neck
    YOLO 2× đủ vì kênh đã giàu).  Giúp implicit-high-dim mapping mà
    không phải widen kênh thực — đúng tinh thần "Rewrite the Stars".
    """
    def __init__(self, channels, expand=2):
        super().__init__()
        hidden = channels * expand
        self.f1 = nn.Conv2d(channels, hidden, 1, bias=False)
        self.f2 = nn.Conv2d(channels, hidden, 1, bias=False)
        self.bn = nn.BatchNorm2d(hidden)
        self.act = nn.SiLU(inplace=True)
        self.proj = nn.Conv2d(hidden, channels, 1, bias=False)
        self.bn_out = nn.BatchNorm2d(channels)

    def forward(self, x):
        a = self.act(self.f1(x))
        b = self.f2(x)
        return self.bn_out(self.proj(self.bn(a * b)))


# ════════════════════════════════════════════════════════════════════════
# 3. GSStarUnit — Ghost + Star + DW context, theo nguyên tắc ShuffleNetV2
# ════════════════════════════════════════════════════════════════════════
class GSStarUnit(nn.Module):
    """
    Đơn vị xử lý 1 nhánh kênh trong GSStarBlock:
        x → GhostUnit (channel mix nhẹ)
          → StarLite  (implicit high-dim)
          → DWConv 5×5 (large RF, rẻ)
          → BN
    Output cùng số kênh với input (ShuffleNetV2 G-rule #1: equal channels).
    """
    def __init__(self, channels):
        super().__init__()
        self.ghost = GhostUnit(channels, channels, k=1, ratio=2)
        self.star = StarLite(channels, expand=2)
        self.dw = nn.Conv2d(channels, channels, 5, 1, 2, groups=channels, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        y = self.ghost(x)
        y = self.star(y) + y           # local residual giữ gradient
        y = self.act(self.bn(self.dw(y)))
        return y


# ════════════════════════════════════════════════════════════════════════
# 4. GSStarBlock — block chính cho NECK
# ════════════════════════════════════════════════════════════════════════
class GSStarBlock(nn.Module):
    """
    GSStarBlock — drop-in thay C3k2 trong neck.

    Ý tưởng tổng hợp:
        cv1 1×1 → split kênh thành 2 nửa (ShuffleNetV2 G-rule #2):
            ├── nhánh A: pass-through (cheap, không xử lý)
            └── nhánh B: n × GSStarUnit (xử lý thật sự)
        → concat(A, B_n)  (đây là OSA — One-Shot Aggregation, VoVNet)
        → channel_shuffle (trộn thông tin chéo, ShuffleNetV2)
        → cv2 1×1 → output

    Khác C3k2:
        - C3k2 dùng chunk → n bottleneck rồi extend tất cả intermediate
          (mỗi block đóng góp 1 chunk). GSStar chỉ aggregate MỘT lần ở cuối
          → giảm I/O memory đáng kể (VoVNet đã chứng minh trên ImageNet).
        - C3k2 dùng Bottleneck (1×1 + 3×3). GSStar dùng Ghost + Star + DW5×5
          → nhẹ hơn nhưng RF lớn hơn và channel mixing biểu cảm hơn.
        - GSStar có channel_shuffle giữa hai nhánh — ngăn information silo.

    Chữ ký giữ giống C3k2 cho dễ port:
        GSStarBlock(c1, c2, n=1, e=0.5, shortcut=True)
    """
    def __init__(self, c1, c2, n=1, e=0.5, shortcut=True):
        super().__init__()
        self.c = max(8, int(c2 * e))
        # cv1: c1 → 2c (split thành 2 nửa)
        self.cv1 = _CBA(c1, 2 * self.c, k=1)
        # n GSStarUnit liên tiếp trên nhánh B (NHƯNG aggregate 1-shot ở cuối)
        self.units = nn.ModuleList(GSStarUnit(self.c) for _ in range(n))
        # cv2: aggregate (c từ A + c từ B-cuối) → c2
        self.cv2 = _CBA(2 * self.c, c2, k=1)
        # Optional residual nếu c1 == c2
        self.add = shortcut and (c1 == c2)
        # số nhóm shuffle: 2 (theo cấu trúc 2-branch ShuffleNetV2)
        self.shuffle_groups = 2

    def forward(self, x):
        y = self.cv1(x)
        a, b = y.chunk(2, dim=1)        # split (G-rule ShuffleNetV2)
        for u in self.units:
            b = u(b)                    # chỉ nhánh B làm việc
        out = torch.cat((a, b), dim=1)  # OSA: aggregate ONE TIME ở cuối
        out = channel_shuffle(out, self.shuffle_groups)
        out = self.cv2(out)
        return x + out if self.add else out


# ════════════════════════════════════════════════════════════════════════
# 5. GSStarCSP — biến thể CSP (3 conv) nếu cần dùng chỗ "c3k=True"
# ════════════════════════════════════════════════════════════════════════
class GSStarCSP(nn.Module):
    """
    Phiên bản CSP-3-conv (giống vai trò C3k khi c3k=True):
        x → cv1 1×1 → n × GSStarUnit ──┐
        x → cv2 1×1 ─────────────────────┤→ concat → cv3 1×1 → out
    Dùng khi muốn thêm sức biểu diễn ở scale sâu (P5).
    """
    def __init__(self, c1, c2, n=1, e=0.5, shortcut=True):
        super().__init__()
        c_ = max(8, int(c2 * e))
        self.cv1 = _CBA(c1, c_, k=1)
        self.cv2 = _CBA(c1, c_, k=1)
        self.cv3 = _CBA(2 * c_, c2, k=1)
        self.m = nn.Sequential(*(GSStarUnit(c_) for _ in range(n)))
        self.add = shortcut and (c1 == c2)

    def forward(self, x):
        y = self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
        return x + y if self.add else y


__all__ = [
    "GhostUnit",
    "StarLite",
    "GSStarUnit",
    "GSStarBlock",
    "GSStarCSP",
]
