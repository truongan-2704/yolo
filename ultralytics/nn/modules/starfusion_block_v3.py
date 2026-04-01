# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
C3k2_DCNF_V3: Supreme StarFusion Block — Version 3 (Optimized)
===============================================================
Pure PyTorch implementation — no torchvision C++ extensions needed.
Works reliably on both CPU and GPU.

Optimizations applied:
1. LightweightDCNv2: Cached base grid via register_buffer → eliminates
   grid recomputation every forward pass (major speedup)
2. Inplace SiLU across all branches → reduced peak memory
3. Fused offset normalization with single cat + permute
4. DualAttentionGate: reuse pooled features, cleaner reshape
5. StarFusionBottleneck_V3: pre-clamped blend_weight with sigmoid
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv


# ===========================================================================
# 1. Lightweight DCNv2 — Pure PyTorch (CPU/GPU compatible, grid-cached)
# ===========================================================================
class LightweightDCNv2(nn.Module):
    """
    Deformable Convolution v2 — Pure PyTorch implementation.
    Uses F.grid_sample to deform input according to learned offsets & mask.
    No torchvision C++ extensions → runs reliably on CPU & GPU.

    Optimization: Base grid is pre-computed and cached as a buffer.
    Only recomputed when spatial dimensions change.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, groups=1):
        super().__init__()
        self.in_channels = in_channels

        # Predict offset (2 channels: dy, dx) + mask (1 channel)
        self.offset_conv = nn.Conv2d(
            in_channels, 3,
            kernel_size=kernel_size, padding=padding, bias=True
        )
        # Init to zero → initially behaves like identity sampling
        nn.init.constant_(self.offset_conv.weight, 0.0)
        nn.init.constant_(self.offset_conv.bias, 0.0)

        # Depthwise conv on deformed features for local mixing
        self.dw_conv = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size, padding=padding,
            groups=in_channels, bias=False
        )
        # Pointwise conv for channel mixing
        self.pw_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)

        # Cache for base grid (avoids recomputation every forward)
        self._cached_h = 0
        self._cached_w = 0
        self.register_buffer('_base_grid', torch.empty(0), persistent=False)

    def _get_base_grid(self, H, W, device, dtype):
        """Get or recompute the base grid, cached for efficiency."""
        if H != self._cached_h or W != self._cached_w or self._base_grid.numel() == 0:
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(-1, 1, H, device=device, dtype=dtype),
                torch.linspace(-1, 1, W, device=device, dtype=dtype),
                indexing='ij'
            )
            self._base_grid = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
            self._cached_h = H
            self._cached_w = W
        return self._base_grid

    def forward(self, x):
        B, C, H, W = x.shape

        # Predict offsets and mask
        offset_mask = self.offset_conv(x)  # [B, 3, H, W]
        offset_y = offset_mask[:, 0:1, :, :]  # [B, 1, H, W]
        offset_x = offset_mask[:, 1:2, :, :]  # [B, 1, H, W]
        mask = torch.sigmoid(offset_mask[:, 2:3, :, :])  # [B, 1, H, W]

        # Get cached base grid
        base_grid = self._get_base_grid(H, W, x.device, x.dtype)
        base_grid = base_grid.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 2]

        # Normalize offsets to [-1, 1] range and build deformed grid
        offset = torch.cat([
            offset_x * (2.0 / W),
            offset_y * (2.0 / H),
        ], dim=1).permute(0, 2, 3, 1)  # [B, H, W, 2]

        deformed_grid = base_grid + offset  # [B, H, W, 2]

        # Sample input at deformed positions
        x_deformed = F.grid_sample(
            x, deformed_grid,
            mode='bilinear', padding_mode='zeros', align_corners=True
        )  # [B, C, H, W]

        # Apply modulation mask
        x_deformed = x_deformed * mask

        # Local + channel mixing
        out = self.dw_conv(x_deformed)
        out = self.pw_conv(out)
        return out


# ===========================================================================
# 2. Dual Attention Gate — Channel (ECA) + Spatial (optimized)
# ===========================================================================
class DualAttentionGate(nn.Module):
    """
    Channel Attention (ECA-style Conv1d) + Spatial Attention (7×7 conv)
    applied sequentially on the star-fused features.
    """

    def __init__(self, channels, eca_kernel=5, spatial_kernel=7):
        super().__init__()
        # --- Channel Attention (ECA) ---
        self.ch_pool = nn.AdaptiveAvgPool2d(1)
        self.ch_conv = nn.Conv1d(
            1, 1, kernel_size=eca_kernel,
            padding=eca_kernel // 2, bias=False
        )
        self.ch_act = nn.Sigmoid()

        # --- Spatial Attention ---
        self.sp_conv = nn.Conv2d(
            2, 1, kernel_size=spatial_kernel,
            padding=spatial_kernel // 2, bias=False
        )
        self.sp_act = nn.Sigmoid()

    def forward(self, x):
        # Channel attention: squeeze spatial → Conv1d → expand
        b, c, _, _ = x.size()
        y_ch = self.ch_pool(x).view(b, 1, c)             # [B, 1, C]
        y_ch = self.ch_act(self.ch_conv(y_ch)).view(b, c, 1, 1)  # [B, C, 1, 1]
        x = x * y_ch

        # Spatial attention: squeeze channel → Conv2d → expand
        avg_sp = x.mean(dim=1, keepdim=True)               # [B, 1, H, W]
        max_sp, _ = x.max(dim=1, keepdim=True)             # [B, 1, H, W]
        sp_desc = torch.cat([avg_sp, max_sp], dim=1)       # [B, 2, H, W]
        y_sp = self.sp_act(self.sp_conv(sp_desc))           # [B, 1, H, W]
        x = x * y_sp

        return x


# ===========================================================================
# 3. StarFusionBottleneck V3 — Supreme Version (Optimized)
# ===========================================================================
class StarFusionBottleneck_V3(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, star_ratio=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels

        # --- Channel reduction ---
        self.cv_reduce = Conv(c1, c_, 1, 1)

        # --- Partial channel split ---
        self.c_star = max(1, int(c_ * star_ratio))
        self.c_bypass = c_ - self.c_star

        # === DCNv2 Branch (Local, shape-aware) ===
        self.branch_dcn = nn.Sequential(
            LightweightDCNv2(self.c_star, self.c_star, kernel_size=3, padding=1, groups=1),
            nn.BatchNorm2d(self.c_star),
            nn.SiLU(inplace=True),
        )

        # === Multi-Scale Context Branches ===
        # Context 1: Dilated d=2, RF = 7×7
        self.ctx_d2 = nn.Sequential(
            nn.Conv2d(self.c_star, self.c_star, kernel_size=3, padding=2,
                      dilation=2, groups=self.c_star, bias=False),
            nn.BatchNorm2d(self.c_star),
            nn.SiLU(inplace=True),
        )
        # Context 2: Dilated d=3, RF = 13×13
        self.ctx_d3 = nn.Sequential(
            nn.Conv2d(self.c_star, self.c_star, kernel_size=3, padding=3,
                      dilation=3, groups=self.c_star, bias=False),
            nn.BatchNorm2d(self.c_star),
            nn.SiLU(inplace=True),
        )
        # Context 3: Asymmetric 1×7 + 7×1
        self.ctx_asym = nn.Sequential(
            nn.Conv2d(self.c_star, self.c_star, kernel_size=(1, 7), padding=(0, 3),
                      groups=self.c_star, bias=False),
            nn.BatchNorm2d(self.c_star),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.c_star, self.c_star, kernel_size=(7, 1), padding=(3, 0),
                      groups=self.c_star, bias=False),
            nn.BatchNorm2d(self.c_star),
            nn.SiLU(inplace=True),
        )

        # Learnable weights for context aggregation
        self.ctx_weight = nn.Parameter(torch.ones(3) / 3.0)

        # === Star Stabilizer ===
        self.star_bn = nn.BatchNorm2d(self.c_star)

        # === Dual Attention Gate ===
        self.dual_gate = DualAttentionGate(self.c_star)

        # === Learnable blend between star path and bypass ===
        self.blend_weight = nn.Parameter(torch.tensor(0.5))

        # === Channel expansion ===
        self.cv_expand = Conv(c_, c2, 1, 1)

        self.add = shortcut and c1 == c2

    @staticmethod
    def _channel_shuffle(x, groups=2):
        """Channel shuffle for cross-group information flow."""
        b, c, h, w = x.shape
        if c % groups != 0:
            return x
        x = x.view(b, groups, c // groups, h, w)
        x = x.transpose(1, 2).contiguous()
        return x.view(b, c, h, w)

    def forward(self, x):
        identity = x
        h = self.cv_reduce(x)

        # Split into star path and bypass
        h_star, h_bypass = h.split([self.c_star, self.c_bypass], dim=1)

        # --- Stage 1: DCN × Context Aggregation ---
        dcn_feat = self.branch_dcn(h_star)

        # Multi-scale context aggregation with learnable weights
        w = torch.softmax(self.ctx_weight, dim=0)
        ctx_agg = (w[0] * self.ctx_d2(h_star) +
                   w[1] * self.ctx_d3(h_star) +
                   w[2] * self.ctx_asym(h_star))

        # ★ Progressive Star: multiplicative fusion (stage 1)
        star = dcn_feat * ctx_agg
        star = self.star_bn(star)

        # --- Stage 2: Dual Attention Gate ---
        star = self.dual_gate(star)

        # --- Concat star + bypass with learnable blend ---
        alpha = torch.sigmoid(self.blend_weight)
        out = torch.cat([star * alpha, h_bypass * (1.0 - alpha)], dim=1)
        out = self._channel_shuffle(out, groups=2)

        # Expand & residual
        out = self.cv_expand(out)
        return out + identity if self.add else out


# ===========================================================================
# 4. C3k2_DCNF_V3 — CSP Wrapper (API-compatible with C3k2)
# ===========================================================================
class C3k2_DCNF_V3(nn.Module):

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        self.c = int(c2 * e)

        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

        self.m = nn.ModuleList(
            StarFusionBottleneck_V3(self.c, self.c, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x):
        """Forward pass: split → n × StarFusionV3 → concat → merge."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
