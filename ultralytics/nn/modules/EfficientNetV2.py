# import torch
# import torch.nn as nn
#
#
# def autopad(k, p=None, d=1):  # kernel, padding, dilation
#     """Pad to 'same' shape outputs."""
#     if d > 1:
#         k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
#     if p is None:
#         p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
#     return p
#
#
# class SE(nn.Module):
#     """Squeeze-and-Excitation Block."""
#
#     def __init__(self, c, r=4):  # c: input channels, r: reduction ratio
#         super().__init__()
#         c_reduced = max(1, c // r)
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Conv2d(c, c_reduced, 1, bias=True),
#             nn.SiLU(inplace=True),
#             nn.Conv2d(c_reduced, c, 1, bias=True),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         return x * self.fc(self.avg_pool(x))
#
#
# class FusedMBConvBlock(nn.Module):
#     """Single Fused-MBConv Block (Unit)."""
#
#     def __init__(self, c1, c2, s=1, expand=4):
#         super().__init__()
#         hidden_c = int(c1 * expand)
#         self.use_res_connect = (s == 1 and c1 == c2)
#
#         if expand == 1:
#             # Nếu không expand, chỉ cần 1 lớp Conv 3x3
#             self.block = nn.Sequential(
#                 nn.Conv2d(c1, c2, 3, s, autopad(3), bias=False),
#                 nn.BatchNorm2d(c2),
#                 nn.SiLU(inplace=True)
#             )
#         else:
#             # Fused: Conv3x3 (Expansion) -> Conv1x1 (Projection)
#             self.block = nn.Sequential(
#                 # Conv 3x3 kết hợp Expansion
#                 nn.Conv2d(c1, hidden_c, 3, s, autopad(3), bias=False),
#                 nn.BatchNorm2d(hidden_c),
#                 nn.SiLU(inplace=True),
#                 # Pointwise Conv 1x1 (Projection)
#                 nn.Conv2d(hidden_c, c2, 1, 1, 0, bias=False),
#                 nn.BatchNorm2d(c2)
#                 # Lưu ý: FusedMBConv thường không dùng SE
#             )
#
#     def forward(self, x):
#         return x + self.block(x) if self.use_res_connect else self.block(x)
#
#
# class MBConvBlock(nn.Module):
#     """Single MBConv Block (Unit)."""
#
#     def __init__(self, c1, c2, s=1, expand=4):
#         super().__init__()
#         hidden_c = int(c1 * expand)
#         self.use_res_connect = (s == 1 and c1 == c2)
#
#         self.block = nn.Sequential(
#             # 1. Expansion (1x1 Conv)
#             nn.Conv2d(c1, hidden_c, 1, 1, 0, bias=False) if expand != 1 else nn.Identity(),
#             nn.BatchNorm2d(hidden_c) if expand != 1 else nn.Identity(),
#             nn.SiLU(inplace=True) if expand != 1 else nn.Identity(),
#
#             # 2. Depthwise Conv (3x3)
#             nn.Conv2d(hidden_c, hidden_c, 3, s, autopad(3), groups=hidden_c, bias=False),
#             nn.BatchNorm2d(hidden_c),
#             nn.SiLU(inplace=True),
#
#             # 3. Squeeze-and-Excitation
#             SE(hidden_c, r=4),
#
#             # 4. Pointwise Conv (1x1) - Projection
#             nn.Conv2d(hidden_c, c2, 1, 1, 0, bias=False),
#             nn.BatchNorm2d(c2)
#         )
#
#     def forward(self, x):
#         return x + self.block(x) if self.use_res_connect else self.block(x)
#
#
# # --------------------------------------------------------------------------------
# # CÁC MODULE CHÍNH SẼ GỌI TRONG YAML
# # --------------------------------------------------------------------------------
#
# class FusedMBConv(nn.Module):
#     """Stack of FusedMBConv Blocks."""
#
#     def __init__(self, c1, c2, n=1, s=1, expand=4):
#         super().__init__()
#         self.blocks = nn.ModuleList()
#
#         # Block đầu tiên: Xử lý Stride (s) và thay đổi Channels (c1 -> c2)
#         self.blocks.append(FusedMBConvBlock(c1, c2, s=s, expand=expand))
#
#         # Các block sau (n-1): Stride=1, Channels giữ nguyên (c2 -> c2)
#         for _ in range(n - 1):
#             self.blocks.append(FusedMBConvBlock(c2, c2, s=1, expand=expand))
#
#     def forward(self, x):
#         for blk in self.blocks:
#             x = blk(x)
#         return x
#
#
# class MBConv(nn.Module):
#     """Stack of MBConv Blocks."""
#
#     def __init__(self, c1, c2, n=1, s=1, expand=4):
#         super().__init__()
#         self.blocks = nn.ModuleList()
#
#         # Block đầu tiên: Xử lý Stride (s) và thay đổi Channels (c1 -> c2)
#         self.blocks.append(MBConvBlock(c1, c2, s=s, expand=expand))
#
#         # Các block sau (n-1): Stride=1, Channels giữ nguyên (c2 -> c2)
#         for _ in range(n - 1):
#             self.blocks.append(MBConvBlock(c2, c2, s=1, expand=expand))
#
#     def forward(self, x):
#         for blk in self.blocks:
#             x = blk(x)
#         return x

import torch
import torch.nn as nn


def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


# --------------------------------------------------------------------------------
# DROP PATH (STOCHASTIC DEPTH)
# --------------------------------------------------------------------------------
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # Xử lý cho tensor có số chiều bất kỳ
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


# --------------------------------------------------------------------------------
# SQUEEZE-AND-EXCITATION BLOCK
# --------------------------------------------------------------------------------
class SE(nn.Module):
    """Squeeze-and-Excitation Block (Chuẩn EfficientNet)."""

    # c_in: input channels của khối (để tính tỷ lệ bóp)
    # c_expand: expanded channels (đầu vào thực tế của SE block)
    def __init__(self, c_in, c_expand, se_ratio=0.25):
        super().__init__()
        # Tính squeezed channels dựa trên c_in thay vì c_expand
        c_reduced = max(1, int(c_in * se_ratio))

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c_expand, c_reduced, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_reduced, c_expand, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.avg_pool(x))


# --------------------------------------------------------------------------------
# CÁC BLOCK CƠ SỞ
# --------------------------------------------------------------------------------
class FusedMBConvBlock(nn.Module):
    """Single Fused-MBConv Block (Unit)."""

    def __init__(self, c1, c2, s=1, expand=4, drop_prob=0.0):
        super().__init__()
        hidden_c = int(c1 * expand)
        self.use_res_connect = (s == 1 and c1 == c2)

        # Bổ sung DropPath
        self.drop_path = DropPath(drop_prob) if self.use_res_connect else nn.Identity()

        if expand == 1:
            self.block = nn.Sequential(
                nn.Conv2d(c1, c2, 3, s, autopad(3), bias=False),
                nn.BatchNorm2d(c2),
                nn.SiLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(c1, hidden_c, 3, s, autopad(3), bias=False),
                nn.BatchNorm2d(hidden_c),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden_c, c2, 1, 1, 0, bias=False),
                nn.BatchNorm2d(c2)
            )

    def forward(self, x):
        # Áp dụng DropPath vào nhánh residual
        return x + self.drop_path(self.block(x)) if self.use_res_connect else self.block(x)


class MBConvBlock(nn.Module):
    """Single MBConv Block (Unit)."""

    def __init__(self, c1, c2, s=1, expand=4, drop_prob=0.0):
        super().__init__()
        hidden_c = int(c1 * expand)
        self.use_res_connect = (s == 1 and c1 == c2)

        # Bổ sung DropPath
        self.drop_path = DropPath(drop_prob) if self.use_res_connect else nn.Identity()

        self.block = nn.Sequential(
            nn.Conv2d(c1, hidden_c, 1, 1, 0, bias=False) if expand != 1 else nn.Identity(),
            nn.BatchNorm2d(hidden_c) if expand != 1 else nn.Identity(),
            nn.SiLU(inplace=True) if expand != 1 else nn.Identity(),

            nn.Conv2d(hidden_c, hidden_c, 3, s, autopad(3), groups=hidden_c, bias=False),
            nn.BatchNorm2d(hidden_c),
            nn.SiLU(inplace=True),

            # Sửa lại params truyền vào SE (c1 làm chuẩn, c_expand làm đầu vào thực tế)
            SE(c_in=c1, c_expand=hidden_c, se_ratio=0.25),

            nn.Conv2d(hidden_c, c2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c2)
        )

    def forward(self, x):
        # Áp dụng DropPath vào nhánh residual
        return x + self.drop_path(self.block(x)) if self.use_res_connect else self.block(x)


# --------------------------------------------------------------------------------
# MODULE GỌI TRONG YAML
# --------------------------------------------------------------------------------
class FusedMBConv(nn.Module):
    """Stack of FusedMBConv Blocks."""

    def __init__(self, c1, c2, n=1, s=1, expand=4, drop_prob=0.0):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.blocks.append(FusedMBConvBlock(c1, c2, s=s, expand=expand, drop_prob=drop_prob))
        for _ in range(n - 1):
            self.blocks.append(FusedMBConvBlock(c2, c2, s=1, expand=expand, drop_prob=drop_prob))

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class MBConv(nn.Module):
    """Stack of MBConv Blocks."""

    def __init__(self, c1, c2, n=1, s=1, expand=4, drop_prob=0.0):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.blocks.append(MBConvBlock(c1, c2, s=s, expand=expand, drop_prob=drop_prob))
        for _ in range(n - 1):
            self.blocks.append(MBConvBlock(c2, c2, s=1, expand=expand, drop_prob=drop_prob))

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

