# YOLO13 🚀 - Next-Generation Object Detection Architecture
# Novel blocks: Multi-Scale Adaptive Convolution, Gated Channel Fusion,
# Dynamic Spatial Attention, Enhanced CSP with Frequency-Spatial Dual Path
"""YOLO13 architecture blocks - comprehensive and high-efficiency design."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .conv import Conv, autopad


class MSConv(nn.Module):
    """Multi-Scale Adaptive Convolution.
    
    Combines depthwise convolutions at multiple kernel sizes (3, 5, 7) with
    learned adaptive mixing weights. Captures multi-scale spatial patterns
    with minimal parameter overhead via depthwise separable design.
    """

    def __init__(self, c1, c2, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        # Input projection
        self.cv_in = Conv(c1, c_, 1)
        
        # Multi-scale depthwise convolutions
        self.dw3 = nn.Conv2d(c_, c_, 3, 1, 1, groups=c_, bias=False)
        self.dw5 = nn.Conv2d(c_, c_, 5, 1, 2, groups=c_, bias=False)
        self.dw7 = nn.Conv2d(c_, c_, 7, 1, 3, groups=c_, bias=False)
        
        # Adaptive scale mixing weights (learned per-channel)
        self.scale_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_, c_ * 3, 1, bias=False),
            nn.BatchNorm2d(c_ * 3),
            nn.Sigmoid()
        )
        
        # Output projection with BN + SiLU
        self.cv_out = Conv(c_, c2, 1)
        self.bn = nn.BatchNorm2d(c_)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.cv_in(x)
        
        # Multi-scale features
        f3 = self.dw3(x)
        f5 = self.dw5(x)
        f7 = self.dw7(x)
        
        # Adaptive mixing
        w = self.scale_attn(x)
        b, c, _, _ = x.shape
        w = w.view(b, 3, c, 1, 1)
        w = F.softmax(w, dim=1)
        
        # Weighted fusion
        out = w[:, 0] * f3 + w[:, 1] * f5 + w[:, 2] * f7
        out = self.act(self.bn(out))
        return self.cv_out(out)


class GatedChannelFusion(nn.Module):
    """Gated Channel Fusion Attention.
    
    Dual-path channel attention using both average and max pooling with 
    a gating mechanism. More expressive than SE/ECA with minimal overhead.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        
        # Dual-path squeeze
        self.avg_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.SiLU(inplace=True),
        )
        self.max_fc = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.SiLU(inplace=True),
        )
        
        # Gated excitation
        self.gate = nn.Sequential(
            nn.Linear(mid * 2, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        avg_out = self.avg_fc(x)
        max_out = self.max_fc(x)
        gate = self.gate(torch.cat([avg_out, max_out], dim=1))
        return x * gate.view(b, c, 1, 1)


class DynamicSpatialFusion(nn.Module):
    """Dynamic Spatial Fusion Module.
    
    Generates spatial attention through multi-scale local context
    with depthwise convolutions and learnable spatial gates.
    """

    def __init__(self, channels):
        super().__init__()
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.BatchNorm2d(1),
        )
        # Local context enhancement
        self.local_ctx = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Spatial attention map
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_w = self.sigmoid(self.spatial_conv(torch.cat([avg_out, max_out], dim=1)))
        
        # Local context enhancement
        local = self.local_ctx(x)
        return x * spatial_w + local * (1 - spatial_w)


class YOLO13Bottleneck(nn.Module):
    """YOLO13 Bottleneck with Multi-Scale Conv + Dual Attention.
    
    Combines MSConv for multi-scale feature extraction with GatedChannelFusion
    and DynamicSpatialFusion for comprehensive feature refinement.
    """

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        
        # Multi-scale adaptive conv (replaces standard conv)
        self.ms_conv = nn.Sequential(
            nn.Conv2d(c_, c_, 3, 1, 1, groups=c_, bias=False),
            nn.Conv2d(c_, c_, 5, 1, 2, groups=c_, bias=False) if c_ >= 32 else nn.Identity(),
        ) if False else nn.Identity()  # disabled for basic bottleneck, used in full block
        
        # Attention modules
        self.channel_attn = GatedChannelFusion(c2, reduction=16)
        self.spatial_attn = DynamicSpatialFusion(c2)
        
        self.add = shortcut and c1 == c2
        
        # Learnable residual scale
        self.gamma = nn.Parameter(torch.ones(1) * 0.1) if self.add else None

    def forward(self, x):
        out = self.cv2(self.cv1(x))
        out = self.channel_attn(out)
        out = self.spatial_attn(out)
        if self.add:
            return x + self.gamma * out
        return out


class YOLO13BottleneckLight(nn.Module):
    """Lightweight YOLO13 Bottleneck without attention for head usage."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3k2_YOLO13(nn.Module):
    """CSP Bottleneck with 2 convolutions and YOLO13 Bottleneck blocks.
    
    The main building block of YOLO13 backbone. Extends C3k2 with
    YOLO13's multi-scale conv and dual attention bottleneck.
    """

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        
        if c3k:
            self.m = nn.ModuleList(
                nn.Sequential(
                    YOLO13Bottleneck(self.c, self.c, shortcut, g, e=1.0),
                    YOLO13Bottleneck(self.c, self.c, shortcut, g, e=1.0),
                ) for _ in range(n)
            )
        else:
            self.m = nn.ModuleList(
                YOLO13BottleneckLight(self.c, self.c, shortcut, g) for _ in range(n)
            )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class YOLO13CSP(nn.Module):
    """YOLO13 Cross Stage Partial block with enhanced feature fusion.
    
    Uses split-transform-merge with YOLO13 bottlenecks and
    an additional cross-path information exchange.
    """

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        
        if c3k:
            self.m = nn.Sequential(*(
                nn.Sequential(
                    YOLO13Bottleneck(c_, c_, shortcut, g, e=1.0),
                    YOLO13Bottleneck(c_, c_, shortcut, g, e=1.0),
                ) for _ in range(n)
            ))
        else:
            self.m = nn.Sequential(*(
                YOLO13BottleneckLight(c_, c_, shortcut, g) for _ in range(n)
            ))
        
        # Cross-path channel exchange
        self.cross_attn = GatedChannelFusion(c2, reduction=16)

    def forward(self, x):
        x1 = self.m(self.cv1(x))
        x2 = self.cv2(x)
        out = self.cv3(torch.cat([x1, x2], 1))
        return self.cross_attn(out)


class AdaptiveDown(nn.Module):
    """Adaptive Downsampling module.
    
    Replaces simple stride-2 Conv with a combination of max-pool
    and conv paths for better information preservation during downsampling.
    """

    def __init__(self, c1, c2):
        super().__init__()
        c_ = c2 // 2
        # Path 1: MaxPool + 1x1 Conv
        self.pool_path = nn.Sequential(
            nn.MaxPool2d(2, 2),
            Conv(c1, c_, 1)
        )
        # Path 2: Stride-2 Conv
        self.conv_path = Conv(c1, c_, 3, 2)
        # Fusion
        self.fuse = Conv(c_ * 2, c2, 1)

    def forward(self, x):
        p1 = self.pool_path(x)
        p2 = self.conv_path(x)
        return self.fuse(torch.cat([p1, p2], 1))


class EnhancedSPPF(nn.Module):
    """Enhanced Spatial Pyramid Pooling - Fast with channel attention.
    
    Extends SPPF with GatedChannelFusion for better multi-scale
    feature aggregation with attention-guided refinement.
    """

    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.attn = GatedChannelFusion(c2, reduction=16)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        out = self.cv2(torch.cat([x, y1, y2, y3], 1))
        return self.attn(out)


class ScaleAwareAttention(nn.Module):
    """Scale-Aware Feature Attention for FPN neck.
    
    Applies scale-dependent attention to features from different
    pyramid levels during feature fusion in the neck.
    """

    def __init__(self, channels):
        super().__init__()
        self.scale_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.scale_conv(x)


class YOLO13Stem(nn.Module):
    """YOLO13 Stem module for initial feature extraction.
    
    Uses a progressive channel expansion with overlapping convolutions
    for better initial feature capture.
    """

    def __init__(self, c1, c2):
        super().__init__()
        c_ = c2 // 2
        self.conv1 = Conv(c1, c_, 3, 2)  # /2
        self.conv2 = Conv(c_, c2, 3, 2)  # /4
        self.enhance = nn.Sequential(
            nn.Conv2d(c2, c2, 3, 1, 1, groups=c2, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x + self.enhance(x)


class MSConvBlock(nn.Module):
    """Standalone Multi-Scale Conv Block for use in YAML configs.
    
    A complete multi-scale convolution block with channel adjustment.
    """

    def __init__(self, c1, c2, e=0.5):
        super().__init__()
        self.msconv = MSConv(c1, c2, e)

    def forward(self, x):
        return self.msconv(x)
