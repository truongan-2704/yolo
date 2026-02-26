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
# class SE(nn.Module):
#     """Squeeze-and-Excitation Block (Chuẩn EfficientNet)."""
#
#     # c_in: input channels của khối (để tính tỷ lệ bóp)
#     # c_expand: expanded channels (đầu vào thực tế của SE block)
#     def __init__(self, c_in, c_expand, se_ratio=0.25):
#         super().__init__()
#         # Tính squeezed channels dựa trên c_in thay vì c_expand
#         c_reduced = max(1, int(c_in * se_ratio))
#
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Conv2d(c_expand, c_reduced, 1, bias=True),
#             nn.SiLU(inplace=True),
#             nn.Conv2d(c_reduced, c_expand, 1, bias=True),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         return x * self.fc(self.avg_pool(x))
class CoordAtt(nn.Module):
    """Coordinate Attention Block (Thay thế hoàn toàn SE)."""

    def __init__(self, c_in, c_expand, reduction=4):
        super().__init__()
        # Tính số kênh bị bóp (mip). Dùng c_expand để giữ lõi Attention đủ to, bắt vật thể nhỏ
        mip = max(8, c_expand // reduction)

        # 1. Tách Pooling ra làm 2 hướng: Chiều dọc (H) và Chiều ngang (W)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        # 2. Khối Conv dùng chung để học mối quan hệ không gian
        self.conv1 = nn.Conv2d(c_expand, mip, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU(inplace=True)  # Dùng SiLU để đồng bộ chuẩn với YOLO11

        # 3. Tách ra lại thành 2 nhánh để tính Attention weight
        self.conv_h = nn.Conv2d(mip, c_expand, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_w = nn.Conv2d(mip, c_expand, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # Ép kiểu 1D pooling
        x_h = self.pool_h(x)  # [N, C, H, 1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # Chuyển vị thành [N, C, W, 1] để dễ Concat

        # Nối 2 hướng lại để học ngữ cảnh chung
        y = torch.cat([x_h, x_w], dim=2)  # Kích thước: [N, C, H+W, 1]
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # Tách trả lại trục X và Y
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # Trả lại định dạng [N, C, 1, W]

        # Đưa qua Sigmoid để ra trọng số (0 -> 1)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        # Nhân chéo tọa độ X và Y vào ảnh gốc
        return identity * a_w * a_h

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

            # Sửa lại params truyền vào CA (c1 làm chuẩn, c_expand làm đầu vào thực tế)
            CoordAtt(c_in=c1, c_expand=hidden_c, reduction=4),

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

