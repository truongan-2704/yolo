# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
YOLO-Prism Ablation Study Modules
==================================

Dedicated ablation modules for YOLO11-Prism experimental study.
This file implements individual component isolations and leave-one-out
ablations without modifying the original `prism_blocks.py`.

Ablated Components:
1. DFDC (Dual-Frequency Decomposed Convolution) -> Replaced with StandardDWConv
2. High-Pass Filter in DFDC                     -> Replaced with DualFreqConv_NoHP
3. MCG (Moment Contrast Gate)                   -> Replaced with nn.Identity()
4. FASR (FreqSpatialRefine in Neck)             -> Replaced with nn.Identity()

Experimental Configurations:
- Baseline:            Standard DWConv, No MCG, No FASR
- Single-Module:       Only DFDC, Only MCG, Only FASR
- Pairwise/Ablated:    w/o FASR (DFDC+MCG), w/o MCG (DFDC+FASR), w/o DFDC (MCG+FASR)
- Sub-module Ablation: w/o High-Pass Filter (DualFreqConv_NoHP + MCG + FASR)
- Full Prism:          DualFreqConv + MomentContrastGate + FreqSpatialRefine
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.prism_blocks import (
    DualFreqConv,
    MomentContrastGate,
    FreqSpatialRefine,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. ABLATION SUB-MODULES
# ─────────────────────────────────────────────────────────────────────────────

class StandardDWConv(nn.Module):
    """
    Standard Depthwise Convolution (Ablation replacement for DFDC).
    
    Processes all spatial frequencies together with a standard DWConv,
    matching the depthwise nature and spatial receptive field of DFDC
    without explicit frequency decomposition.
    """

    def __init__(self, c, k=3):
        super().__init__()
        self.dw = nn.Conv2d(
            c, c, k, stride=1, padding=k // 2, groups=c, bias=False
        )
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.dw(x)))


class DualFreqConv_NoHP(nn.Module):
    """
    Dual-Frequency Conv WITHOUT High-Pass Filter (Ablation module).
    
    Splits channels into two groups and applies two DWConvs of kernels
    k_lo and k_hi on RAW features (no x - AvgPool(x) high-pass filtering).
    This serves as a HeteroConv control to isolate the exact impact of
    the parameter-free high-pass frequency decomposition.
    """

    def __init__(self, c, k_lo=5, k_hi=3):
        super().__init__()
        assert c >= 2, f"DualFreqConv_NoHP requires c >= 2, got c={c}"

        self.c_lo = c // 2
        self.c_hi = c - self.c_lo

        self.dw_lo = nn.Conv2d(
            self.c_lo, self.c_lo, k_lo, stride=1, padding=k_lo // 2,
            groups=self.c_lo, bias=False
        )
        self.dw_hi = nn.Conv2d(
            self.c_hi, self.c_hi, k_hi, stride=1, padding=k_hi // 2,
            groups=self.c_hi, bias=False
        )
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x_lo = x[:, :self.c_lo]
        x_hi = x[:, self.c_lo:]

        y_lo = self.dw_lo(x_lo)
        y_hi = self.dw_hi(x_hi)  # Raw features directly (no high-pass)

        y = torch.cat([y_lo, y_hi], dim=1)
        y = self._channel_shuffle(y, groups=2)
        return self.act(self.bn(y))

    @staticmethod
    def _channel_shuffle(x, groups=2):
        B, C, H, W = x.shape
        if C % groups != 0:
            return x
        return (
            x.view(B, groups, C // groups, H, W)
            .transpose(1, 2)
            .contiguous()
            .view(B, C, H, W)
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. GENERAL ABLATION BOTTLENECK
# ─────────────────────────────────────────────────────────────────────────────

class PrismAblationBottleneck(nn.Module):
    """
    Parametric Bottleneck for YOLO11-Prism Ablation Studies.
    
    Structure:
        1x1 Expand -> Spatial Op -> Channel Gate -> 1x1 Project -> Residual Add
    
    Args:
        c1 (int): Input channels.
        c2 (int): Output channels.
        shortcut (bool): Use residual connection when c1 == c2.
        g (int): Groups.
        k (int): Base kernel scale (3 or 5).
        e (float): Expansion ratio.
        use_dfdc (bool): True -> DualFreqConv, False -> StandardDWConv.
        use_hp (bool): True -> with High-Pass, False -> without High-Pass (DualFreqConv_NoHP).
        use_mcg (bool): True -> MomentContrastGate, False -> nn.Identity().
    """

    def __init__(
        self,
        c1,
        c2,
        shortcut=True,
        g=1,
        k=3,
        e=1.0,
        use_dfdc=True,
        use_hp=True,
        use_mcg=True,
    ):
        super().__init__()
        c_hidden = int(c2 * e)

        # 1x1 Expand
        self.expand_conv = Conv(c1, c_hidden, 1)

        # Spatial Processing module
        if use_dfdc:
            if k >= 5:
                k_lo, k_hi, hp_k = 7, 5, 5
            else:
                k_lo, k_hi, hp_k = 5, 3, 3

            if use_hp:
                self.spatial_op = DualFreqConv(c_hidden, k_lo=k_lo, k_hi=k_hi, hp_k=hp_k)
            else:
                self.spatial_op = DualFreqConv_NoHP(c_hidden, k_lo=k_lo, k_hi=k_hi)
        else:
            k_std = 5 if k >= 5 else 3
            self.spatial_op = StandardDWConv(c_hidden, k=k_std)

        # Channel Attention module
        if use_mcg:
            self.gate = MomentContrastGate(c_hidden)
        else:
            self.gate = nn.Identity()

        # 1x1 Project
        self.project = Conv(c_hidden, c2, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.expand_conv(x)
        y = self.spatial_op(y)
        y = self.gate(y)
        y = self.project(y)
        return x + y if self.add else y


# ─────────────────────────────────────────────────────────────────────────────
# 3. BASE CONTAINER (C2f / CSP)
# ─────────────────────────────────────────────────────────────────────────────

class _C3k2_PrismAblationContainer(nn.Module):
    """Base class for C2f split-concat containers in Prism ablation studies."""

    def __init__(
        self,
        c1,
        c2,
        n=1,
        c3k=False,
        e=0.5,
        g=1,
        shortcut=True,
        use_dfdc=True,
        use_hp=True,
        use_mcg=True,
        use_fasr=False,
    ):
        super().__init__()
        self.c = int(c2 * e)
        k = 5 if c3k else 3

        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            PrismAblationBottleneck(
                self.c,
                self.c,
                shortcut=shortcut,
                g=g,
                k=k,
                use_dfdc=use_dfdc,
                use_hp=use_hp,
                use_mcg=use_mcg,
            )
            for _ in range(n)
        )
        self.fsr = FreqSpatialRefine(c2) if use_fasr else nn.Identity()

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        return self.fsr(out)

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        return self.fsr(out)


# ─────────────────────────────────────────────────────────────────────────────
# 4. EXPLICIT ABLATION CLASSES FOR BACKBONE AND NECK
# ─────────────────────────────────────────────────────────────────────────────

# --- Configuration 0: Baseline (Standard DWConv, No MCG, No FASR) ---
class C3k2_Prism_Base(_C3k2_PrismAblationContainer):
    """Baseline Backbone container: Standard DWConv, No MCG, No FASR."""
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut,
                         use_dfdc=False, use_hp=False, use_mcg=False, use_fasr=False)


class PrismCSP_Base(_C3k2_PrismAblationContainer):
    """Baseline Neck container: Standard DWConv, No MCG, No FASR."""
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut,
                         use_dfdc=False, use_hp=False, use_mcg=False, use_fasr=False)


# --- Configuration 1: DFDC Only (DFDC active, No MCG, No FASR) ---
class C3k2_Prism_DFDC(_C3k2_PrismAblationContainer):
    """Backbone container: DFDC only (No MCG)."""
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut,
                         use_dfdc=True, use_hp=True, use_mcg=False, use_fasr=False)


class PrismCSP_DFDC(_C3k2_PrismAblationContainer):
    """Neck container: DFDC only (No MCG, No FASR)."""
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut,
                         use_dfdc=True, use_hp=True, use_mcg=False, use_fasr=False)


# --- Configuration 2: MCG Only (Standard DWConv, MCG active, No FASR) ---
class C3k2_Prism_MCG(_C3k2_PrismAblationContainer):
    """Backbone container: MCG only (Standard DWConv, No DFDC)."""
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut,
                         use_dfdc=False, use_hp=False, use_mcg=True, use_fasr=False)


class PrismCSP_MCG(_C3k2_PrismAblationContainer):
    """Neck container: MCG only (Standard DWConv, No DFDC, No FASR)."""
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut,
                         use_dfdc=False, use_hp=False, use_mcg=True, use_fasr=False)


# --- Configuration 3: FASR Only (Standard DWConv, No MCG, FASR active in Neck) ---
class PrismCSP_FASR(_C3k2_PrismAblationContainer):
    """Neck container: FASR only (Standard DWConv, No DFDC, No MCG)."""
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut,
                         use_dfdc=False, use_hp=False, use_mcg=False, use_fasr=True)


# --- Configuration 4: w/o FASR (DFDC + MCG active, FASR ablated in Neck) ---
class PrismCSP_NoFASR(_C3k2_PrismAblationContainer):
    """Neck container: DFDC + MCG, WITHOUT FASR (Ablation w/o FASR)."""
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut,
                         use_dfdc=True, use_hp=True, use_mcg=True, use_fasr=False)


# --- Configuration 5: w/o MCG (DFDC + FASR active, MCG ablated) ---
class PrismCSP_DFDC_FASR(_C3k2_PrismAblationContainer):
    """Neck container: DFDC + FASR, WITHOUT MCG (Ablation w/o MCG)."""
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut,
                         use_dfdc=True, use_hp=True, use_mcg=False, use_fasr=True)


# --- Configuration 6: w/o DFDC (MCG + FASR active, DFDC ablated to DWConv) ---
class PrismCSP_MCG_FASR(_C3k2_PrismAblationContainer):
    """Neck container: MCG + FASR, WITHOUT DFDC (Ablation w/o DFDC)."""
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut,
                         use_dfdc=False, use_hp=False, use_mcg=True, use_fasr=True)


# --- Configuration 7: w/o High-Pass Filter (DualFreqConv_NoHP + MCG + FASR) ---
class C3k2_Prism_NoHP(_C3k2_PrismAblationContainer):
    """Backbone container: DFDC without High-Pass filter + MCG."""
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut,
                         use_dfdc=True, use_hp=False, use_mcg=True, use_fasr=False)


class PrismCSP_NoHP(_C3k2_PrismAblationContainer):
    """Neck container: DFDC without High-Pass filter + MCG + FASR."""
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut,
                         use_dfdc=True, use_hp=False, use_mcg=True, use_fasr=True)


__all__ = [
    "StandardDWConv",
    "DualFreqConv_NoHP",
    "PrismAblationBottleneck",
    "C3k2_Prism_Base",
    "PrismCSP_Base",
    "C3k2_Prism_DFDC",
    "PrismCSP_DFDC",
    "C3k2_Prism_MCG",
    "PrismCSP_MCG",
    "PrismCSP_FASR",
    "PrismCSP_NoFASR",
    "PrismCSP_DFDC_FASR",
    "PrismCSP_MCG_FASR",
    "C3k2_Prism_NoHP",
    "PrismCSP_NoHP",
]
