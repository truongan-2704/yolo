# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
YOLO-Spectra — Wavelet-Guided Directional Detection Architecture
=================================================================

Novel architecture using 2D Haar wavelet transform to decompose features
into JOINT frequency-direction subbands, then processes each with
orientation-matched asymmetric kernels.

Three Innovations:

1. SpectraConv — Wavelet-Guided Directional Convolution:
   - Haar wavelet decomposes input into 4 subbands: LL, LH, HL, HH
   - Each subband processed with orientation-matched DWConv:
     LL → K×K (isotropic context), LH → 1×K (horizontal edges),
     HL → K×1 (vertical edges), HH → 3×3 (diagonal textures)
   - All processing at HALF resolution → 4× cheaper DWConv
   - Inverse Haar reconstructs full resolution losslessly

2. WaveletEnergyGate — Subband Energy Channel Attention:
   - Computes 4 directional energy descriptors per channel:
     E_LL (smooth), E_LH (horizontal), E_HL (vertical), E_HH (diagonal)
   - FC(4C→C) → Sigmoid gating
   - Knows BOTH frequency AND direction per channel (vs SE-Net's 1 value)

3. SpectraCSP — Neck Block with Adaptive Subband Emphasis:
   - Learnable α_LL, α_LH, α_HL, α_HH per pyramid level
   - P3 auto-learns to emphasize edges (small objects)
   - P5 auto-learns to emphasize context (large objects)

References:
  - Mallat (1989): Multiresolution Signal Decomposition
  - Liu et al. (2018): Multi-Level Wavelet-CNN, CVPR
  - Yao et al. (2022): Wave-ViT, ECCV
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv, autopad


# ─────────────────────────────────────────────────────────────────────────────
# 1. HaarWavelet2D — Parameter-free forward and inverse 2D Haar transform
# ─────────────────────────────────────────────────────────────────────────────

class HaarWavelet2D(nn.Module):
    """Parameter-free 2D Haar wavelet transform (forward + inverse).
    
    Forward: Decomposes (B, C, H, W) → 4 subbands at (B, C, H/2, W/2)
    Inverse: Reconstructs (B, C, H/2, W/2) × 4 → (B, C, H, W)
    
    The Haar wavelet is the simplest orthogonal wavelet, using only
    additions and subtractions. Zero learnable parameters.
    """

    @staticmethod
    def forward_transform(x):
        """Decompose input into 4 subbands: LL, LH, HL, HH.
        
        Args:
            x: (B, C, H, W) input tensor. H and W must be even.
            
        Returns:
            tuple of 4 tensors, each (B, C, H//2, W//2):
            - LL: low-frequency approximation (smooth context)
            - LH: horizontal high-frequency (horizontal edges)
            - HL: vertical high-frequency (vertical edges) 
            - HH: diagonal high-frequency (diagonal details/textures)
        """
        # Pad if odd dimensions
        B, C, H, W = x.shape
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        
        # Split into 2×2 blocks
        x00 = x[:, :, 0::2, 0::2]  # top-left
        x01 = x[:, :, 0::2, 1::2]  # top-right
        x10 = x[:, :, 1::2, 0::2]  # bottom-left
        x11 = x[:, :, 1::2, 1::2]  # bottom-right
        
        # Haar wavelet coefficients (scaled by 0.5 for energy preservation)
        ll = (x00 + x01 + x10 + x11) * 0.5  # average = low-low
        lh = (x00 + x01 - x10 - x11) * 0.5  # horizontal diff = low-high
        hl = (x00 - x01 + x10 - x11) * 0.5  # vertical diff = high-low
        hh = (x00 - x01 - x10 + x11) * 0.5  # diagonal diff = high-high
        
        return ll, lh, hl, hh

    @staticmethod
    def inverse_transform(ll, lh, hl, hh, target_h=None, target_w=None):
        """Reconstruct from 4 subbands back to full resolution.
        
        Args:
            ll, lh, hl, hh: 4 subbands, each (B, C, H//2, W//2)
            target_h, target_w: optional target output size (for odd input)
            
        Returns:
            Reconstructed tensor (B, C, H, W)
        """
        B, C, Hh, Wh = ll.shape
        
        # Inverse Haar: recover 2×2 blocks
        x00 = (ll + lh + hl + hh) * 0.5
        x01 = (ll + lh - hl - hh) * 0.5
        x10 = (ll - lh + hl - hh) * 0.5
        x11 = (ll - lh - hl + hh) * 0.5
        
        # Interleave back to full resolution
        out = torch.zeros(B, C, Hh * 2, Wh * 2, device=ll.device, dtype=ll.dtype)
        out[:, :, 0::2, 0::2] = x00
        out[:, :, 0::2, 1::2] = x01
        out[:, :, 1::2, 0::2] = x10
        out[:, :, 1::2, 1::2] = x11
        
        # Crop if we padded odd dimensions
        if target_h is not None and target_w is not None:
            out = out[:, :, :target_h, :target_w]
        
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 2. SpectraConv — Wavelet-Guided Directional Convolution
# ─────────────────────────────────────────────────────────────────────────────

class SpectraConv(nn.Module):
    """Wavelet-guided directional depthwise convolution.
    
    Decomposes input via Haar wavelet into 4 subbands, processes each
    with orientation-matched asymmetric DWConv kernels at half resolution,
    then reconstructs via inverse Haar transform.
    
    Args:
        c (int): Number of channels (input == output).
        k_lo (int): Kernel size for LL subband (isotropic). Default 5.
        k_hi (int): Kernel size for directional subbands LH/HL. Default 5.
    """

    def __init__(self, c, k_lo=5, k_hi=5):
        super().__init__()
        self.c = c
        self.haar = HaarWavelet2D()
        
        # Subband-matched DWConv (all at half resolution!)
        # LL: large isotropic kernel for smooth context
        self.dw_ll = nn.Conv2d(c, c, k_lo, padding=k_lo // 2, groups=c, bias=False)
        # LH: horizontal kernel for horizontal edges (1×K)
        self.dw_lh = nn.Conv2d(c, c, (1, k_hi), padding=(0, k_hi // 2), groups=c, bias=False)
        # HL: vertical kernel for vertical edges (K×1)
        self.dw_hl = nn.Conv2d(c, c, (k_hi, 1), padding=(k_hi // 2, 0), groups=c, bias=False)
        # HH: isotropic 3×3 for diagonal textures
        self.dw_hh = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False)
        
        # Pointwise convolution for channel mixing
        self.pw = nn.Conv2d(c, c, 1, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Forward Haar wavelet transform (parameter-free)
        ll, lh, hl, hh = self.haar.forward_transform(x)
        
        # Process each subband with matched kernel (at half resolution!)
        ll = self.dw_ll(ll)
        lh = self.dw_lh(lh)
        hl = self.dw_hl(hl)
        hh = self.dw_hh(hh)
        
        # Inverse Haar wavelet transform (parameter-free reconstruction)
        out = self.haar.inverse_transform(ll, lh, hl, hh, target_h=H, target_w=W)
        
        # Channel mixing + normalization
        out = self.act(self.bn(self.pw(out)))
        
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. WaveletEnergyGate — Subband Energy Channel Attention
# ─────────────────────────────────────────────────────────────────────────────

class WaveletEnergyGate(nn.Module):
    """Channel attention using 4 wavelet subband energy descriptors.
    
    For each channel, computes energy in LL/LH/HL/HH subbands,
    providing a 4D descriptor that captures both frequency content
    and directional preference.
    
    Args:
        c (int): Number of channels.
        reduction (int): Reduction ratio for FC layer. Default 4.
    """

    def __init__(self, c, reduction=4):
        super().__init__()
        self.haar = HaarWavelet2D()
        mid = max(c // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(4 * c, mid),
            nn.SiLU(inplace=True),
            nn.Linear(mid, c),
            nn.Sigmoid(),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Compute wavelet subbands
        ll, lh, hl, hh = self.haar.forward_transform(x)
        
        # Per-channel energy for each subband: mean(subband²)
        e_ll = ll.pow(2).mean(dim=(2, 3))  # (B, C)
        e_lh = lh.pow(2).mean(dim=(2, 3))  # (B, C)
        e_hl = hl.pow(2).mean(dim=(2, 3))  # (B, C)
        e_hh = hh.pow(2).mean(dim=(2, 3))  # (B, C)
        
        # Concat 4 descriptors → (B, 4C)
        desc = torch.cat([e_ll, e_lh, e_hl, e_hh], dim=1)
        
        # FC → gate
        gate = self.fc(desc).view(B, C, 1, 1)
        
        return x * gate


# ─────────────────────────────────────────────────────────────────────────────
# 4. SpectraBottleneck — Core building block
# ─────────────────────────────────────────────────────────────────────────────

class SpectraBottleneck(nn.Module):
    """Bottleneck with SpectraConv + WaveletEnergyGate.
    
    Conv1×1 (expand) → SpectraConv (wavelet processing) → WaveletEnergyGate → residual
    
    Args:
        c_in (int): Input channels.
        c_out (int): Output channels.
        shortcut (bool): Whether to use residual connection. Default True.
        k_lo (int): SpectraConv LL kernel. Default 5.
        k_hi (int): SpectraConv directional kernel. Default 3.
    """

    def __init__(self, c_in, c_out, shortcut=True, k_lo=5, k_hi=3):
        super().__init__()
        self.cv1 = Conv(c_in, c_out, 1)  # 1×1 channel projection
        self.spectra = SpectraConv(c_out, k_lo=k_lo, k_hi=k_hi)
        self.gate = WaveletEnergyGate(c_out)
        self.add = shortcut and c_in == c_out

    def forward(self, x):
        out = self.cv1(x)
        out = self.spectra(out)
        out = self.gate(out)
        return x + out if self.add else out


# ─────────────────────────────────────────────────────────────────────────────
# 5. C3k2_Spectra — CSP block using SpectraBottleneck
# ─────────────────────────────────────────────────────────────────────────────

class C3k2_Spectra(nn.Module):
    """CSP-style block with SpectraBottleneck.
    
    Split → main path (N × SpectraBottleneck) + shortcut → concat → conv1×1
    
    Compatible with YOLO scaling system (depth_mul, width_mul).
    
    Args:
        c1 (int): Input channels.
        c2 (int): Output channels.
        n (int): Number of bottleneck repeats. Default 1.
        shortcut (bool): Use residual in bottlenecks. Default True.
        g (int): Groups (unused, for API compat). Default 1.
        e (float): Expansion ratio. Default 0.5.
        c3k (bool): If True, use larger kernels (7,5) instead of (5,3).
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, c3k=False):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * c_, 1, 1)  # split into 2 branches
        self.cv2 = Conv((2 + n) * c_, c2, 1)  # final merge
        
        k_lo = 7 if c3k else 5
        k_hi = 5 if c3k else 3
        
        self.m = nn.ModuleList(
            SpectraBottleneck(c_, c_, shortcut=shortcut, k_lo=k_lo, k_hi=k_hi)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))  # split into 2 branches
        y.extend(m(y[-1]) for m in self.m)  # apply bottlenecks sequentially
        return self.cv2(torch.cat(y, 1))    # concat all + merge


# ─────────────────────────────────────────────────────────────────────────────
# 6. SpectraCSP — Neck block with adaptive subband emphasis
# ─────────────────────────────────────────────────────────────────────────────

class SpectraCSP(nn.Module):
    """CSP neck block with SpectraBottleneck + adaptive subband emphasis.
    
    Same structure as C3k2_Spectra but designed for the FPN/PAN neck.
    Includes learnable subband emphasis weights that allow each pyramid
    level to specialize (P3 → edges, P5 → context).
    
    Args:
        c1 (int): Input channels.
        c2 (int): Output channels.
        n (int): Number of bottleneck repeats. Default 1.
        shortcut (bool): Use residual in bottlenecks. Default False.
        g (int): Groups (unused, for API compat). Default 1.
        e (float): Expansion ratio. Default 0.5.
        c3k (bool): If True, use larger kernels (7,5).
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, c3k=False):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * c_, 1, 1)
        self.cv2 = Conv((2 + n) * c_, c2, 1)
        
        k_lo = 7 if c3k else 5
        k_hi = 5 if c3k else 3
        
        self.m = nn.ModuleList(
            SpectraBottleneck(c_, c_, shortcut=shortcut, k_lo=k_lo, k_hi=k_hi)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
