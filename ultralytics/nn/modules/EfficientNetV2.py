import torch
import torch.nn as nn


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class SE(nn.Module):
    """Squeeze-and-Excitation Block."""

    def __init__(self, c, r=4):  # c: input channels, r: reduction ratio
        super().__init__()
        c_reduced = max(1, c // r)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c, c_reduced, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_reduced, c, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.avg_pool(x))


class FusedMBConvBlock(nn.Module):
    """Single Fused-MBConv Block (Unit)."""

    def __init__(self, c1, c2, s=1, expand=4):
        super().__init__()
        hidden_c = int(c1 * expand)
        self.use_res_connect = (s == 1 and c1 == c2)

        if expand == 1:
            # Nếu không expand, chỉ cần 1 lớp Conv 3x3
            self.block = nn.Sequential(
                nn.Conv2d(c1, c2, 3, s, autopad(3), bias=False),
                nn.BatchNorm2d(c2),
                nn.SiLU(inplace=True)
            )
        else:
            # Fused: Conv3x3 (Expansion) -> Conv1x1 (Projection)
            self.block = nn.Sequential(
                # Conv 3x3 kết hợp Expansion
                nn.Conv2d(c1, hidden_c, 3, s, autopad(3), bias=False),
                nn.BatchNorm2d(hidden_c),
                nn.SiLU(inplace=True),
                # Pointwise Conv 1x1 (Projection)
                nn.Conv2d(hidden_c, c2, 1, 1, 0, bias=False),
                nn.BatchNorm2d(c2)
                # Lưu ý: FusedMBConv thường không dùng SE
            )

    def forward(self, x):
        return x + self.block(x) if self.use_res_connect else self.block(x)


class MBConvBlock(nn.Module):
    """Single MBConv Block (Unit)."""

    def __init__(self, c1, c2, s=1, expand=4):
        super().__init__()
        hidden_c = int(c1 * expand)
        self.use_res_connect = (s == 1 and c1 == c2)

        self.block = nn.Sequential(
            # 1. Expansion (1x1 Conv)
            nn.Conv2d(c1, hidden_c, 1, 1, 0, bias=False) if expand != 1 else nn.Identity(),
            nn.BatchNorm2d(hidden_c) if expand != 1 else nn.Identity(),
            nn.SiLU(inplace=True) if expand != 1 else nn.Identity(),

            # 2. Depthwise Conv (3x3)
            nn.Conv2d(hidden_c, hidden_c, 3, s, autopad(3), groups=hidden_c, bias=False),
            nn.BatchNorm2d(hidden_c),
            nn.SiLU(inplace=True),

            # 3. Squeeze-and-Excitation
            SE(hidden_c, r=4),

            # 4. Pointwise Conv (1x1) - Projection
            nn.Conv2d(hidden_c, c2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c2)
        )

    def forward(self, x):
        return x + self.block(x) if self.use_res_connect else self.block(x)


# --------------------------------------------------------------------------------
# CÁC MODULE CHÍNH SẼ GỌI TRONG YAML
# --------------------------------------------------------------------------------

class FusedMBConv(nn.Module):
    """Stack of FusedMBConv Blocks."""

    def __init__(self, c1, c2, n=1, s=1, expand=4):
        super().__init__()
        self.blocks = nn.ModuleList()

        # Block đầu tiên: Xử lý Stride (s) và thay đổi Channels (c1 -> c2)
        self.blocks.append(FusedMBConvBlock(c1, c2, s=s, expand=expand))

        # Các block sau (n-1): Stride=1, Channels giữ nguyên (c2 -> c2)
        for _ in range(n - 1):
            self.blocks.append(FusedMBConvBlock(c2, c2, s=1, expand=expand))

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class MBConv(nn.Module):
    """Stack of MBConv Blocks."""

    def __init__(self, c1, c2, n=1, s=1, expand=4):
        super().__init__()
        self.blocks = nn.ModuleList()

        # Block đầu tiên: Xử lý Stride (s) và thay đổi Channels (c1 -> c2)
        self.blocks.append(MBConvBlock(c1, c2, s=s, expand=expand))

        # Các block sau (n-1): Stride=1, Channels giữ nguyên (c2 -> c2)
        for _ in range(n - 1):
            self.blocks.append(MBConvBlock(c2, c2, s=1, expand=expand))

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x