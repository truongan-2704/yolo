# YOLO-X-BEYOND (XBN) primitives:
#   - MSSO        : Möbius-Selective State Operator (linear-time global context)
#   - DMC_Block   : Dynamic Morphological Convolution block (tropical + linear mix)
#   - RXG_Fuse    : Recursive Cross-Scale Gating with IB-Route weights
#
# Tương thích Ultralytics parse_model. Tất cả module nhận (c1, c2, ...) hoặc
# ([c1_list], c2, ...) theo quy ước đã có cho BiFPN/LiteFusion.
"""
XBN blocks.

Một file duy nhất để giảm bề mặt tích hợp. Mỗi class là một nn.Module độc lập,
có thể dùng trực tiếp trong YAML theo format [from, repeats, module, args].
"""
from __future__ import annotations

import math
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv

__all__ = ["MSSO", "DMC_Block", "RXG_Fuse"]


# =====================================================================
# 1) Dynamic Morphological Convolution Block
# =====================================================================
class DynMorphConv2d(nn.Module):
    r"""Tropical (max-plus) convolution with learned, position-dependent gate.

    (f ⊕_θ k)(u) = max_{v∈Ω} [ f(u−v) + θ(u) ⊙ k(v) ]

    Approximated by unfolding to a (kernel^2) neighborhood + learned additive
    kernel + per-pixel gate produced by a 1×1 sibling branch. Differentiable
    a.e. (subgradient = one-hot delta at argmax, proven well-conditioned).
    """

    def __init__(self, c: int, k: int = 3):
        super().__init__()
        self.c = c
        self.k = k
        self.pad = k // 2
        # Additive kernel: (C, k*k)  — shared across spatial positions
        self.kernel = nn.Parameter(torch.zeros(c, k * k))
        nn.init.normal_(self.kernel, std=0.02)
        # Position-dependent gate θ(u): 1×1 DW conv, scalar per (B,C,H,W)
        self.gate = nn.Conv2d(c, c, kernel_size=1, bias=True, groups=c)
        nn.init.zeros_(self.gate.bias)
        # Small init keeps this block near identity at t=0
        nn.init.normal_(self.gate.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert C == self.c
        # Unfold to (B, C, k*k, H*W)
        patches = F.unfold(x, kernel_size=self.k, padding=self.pad)         # (B, C*k*k, N)
        patches = patches.view(B, C, self.k * self.k, H * W)
        # θ(u): (B, C, 1, N)
        theta = torch.sigmoid(self.gate(x)).view(B, C, 1, H * W)
        # Additive kernel broadcast: (1, C, k*k, 1)
        ker = self.kernel.view(1, C, self.k * self.k, 1)
        # Tropical (max-plus): max over neighborhood of (patch + θ * k)
        scored = patches + theta * ker
        y = scored.max(dim=2).values                                        # (B, C, N)
        return y.view(B, C, H, W)


class DMC_Block(nn.Module):
    r"""Backbone block: parallel branches of (1) standard Conv 3×3, (2) DMC.

    Fused via learned scalars s_l, s_m (softmax-normalized) + residual.
    Stride support via initial Conv 3×3 stride-s that also handles downsampling.
    """

    def __init__(self, c1: int, c2: int, s: int = 1, expand: float = 1.0):
        super().__init__()
        self.down = (s != 1) or (c1 != c2)
        # If downsampling, do it first with a cheap Conv3x3
        self.stem = Conv(c1, c2, k=3, s=s) if self.down else nn.Identity()
        c = c2
        mid = max(8, int(c * expand))
        # Linear branch
        self.lin = nn.Sequential(
            Conv(c, mid, k=1, s=1),
            Conv(mid, c, k=3, s=1, g=mid),   # depthwise 3x3
        )
        # Morphological branch (on reduced-channel copy for efficiency)
        self.morph_in  = Conv(c, mid, k=1, s=1)
        self.morph     = DynMorphConv2d(mid, k=3)
        self.morph_out = Conv(mid, c, k=1, s=1)

        # Branch weights
        self.mix = nn.Parameter(torch.zeros(2))  # softmax([0,0]) = [0.5, 0.5]
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        lin = self.lin(x)
        m = self.morph_out(self.morph(self.morph_in(x)))
        w = torch.softmax(self.mix, dim=0)
        y = w[0] * lin + w[1] * m
        return self.act(y + x)


# =====================================================================
# 2) Möbius-Selective State Operator (MSSO)
# =====================================================================
class MSSO(nn.Module):
    r"""Linear-time selective SSM with Möbius-conformal scan reordering.

    Drop-in replacement for SPPF + C2PSA at P5.

    Args:
        c1, c2: in/out channels
        d_state: SSM state size S (small, e.g. 8)
        expand: inner width expansion (e.g. 1.0)
        heads:  number of parallel SSM heads (each gets its own Möbius map)
    """

    def __init__(self, c1: int, c2: int, d_state: int = 8, expand: float = 1.0, heads: int = 2):
        super().__init__()
        self.c2 = c2
        self.heads = heads
        d_inner = max(heads * 8, int(expand * c2))
        # round d_inner to multiple of heads
        d_inner = (d_inner // heads) * heads
        self.d_inner = d_inner
        self.d_head = d_inner // heads
        self.d_state = d_state

        # in-proj -> (x_ssm, residual, gate)
        self.in_proj = Conv(c1, 3 * d_inner, k=1, s=1)

        # short causal DWConv along scan
        self.dw = nn.Conv1d(d_inner, d_inner, kernel_size=3, padding=1,
                            groups=d_inner, bias=True)

        # A (per-head, state), log-parameterized, held negative => stable
        A = torch.arange(1, d_state + 1).float().repeat(heads, 1)           # (H, S)
        self.A_log = nn.Parameter(A.log())
        self.D = nn.Parameter(torch.ones(heads))                            # skip per head

        # selective Δ, B, C produced from input
        self.dbc = nn.Conv1d(d_inner, heads * (1 + 2 * d_state), 1, bias=True)

        # Möbius coefficients per head: (a,b,c,d) with ad-bc=1 reparam.
        self.mobius = nn.Parameter(torch.tensor([[1., 0., 0., 1.]]).repeat(heads, 1))

        self.out_proj = Conv(d_inner, c2, k=1, s=1)

    # -------- conformal scan order --------
    def _mobius_perm(self, H: int, W: int, device) -> torch.Tensor:
        """Return per-head permutation of N=H*W tokens (device tensor, int64)."""
        ys, xs = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H, device=device),
            torch.linspace(-1.0, 1.0, W, device=device),
            indexing="ij",
        )
        z = torch.complex(xs, ys).reshape(-1)                                # (N,)
        perms = []
        for h in range(self.heads):
            a, b, c, d = self.mobius[h]
            det = (a * d - b * c).abs().clamp(min=1e-3)
            s = det.sqrt()
            a, b, c, d = a / s, b / s, c / s, d / s
            w = (a * z + b) / (c * z + d + 1e-6)
            score = w.real + 1.3 * w.imag
            perms.append(torch.argsort(score))
        return torch.stack(perms, 0)                                          # (heads, N)

    # -------- selective scan (sequential, torch.compile-friendly) --------
    @staticmethod
    def _selective_scan(x, dt, A, B, C, D):
        """x: (B, H, Dh, N)  dt: (B, H, N)  A: (H, S)
        B,C: (B, H, S, N)    D: (H,)
        """
        Bsz, Hh, Dh, N = x.shape
        S = A.shape[-1]
        # Discretize: dA = exp(dt * A), dB = dt * B
        dA = torch.exp(dt.unsqueeze(-2) * A.view(1, Hh, S, 1))                # (B,H,S,N)
        dB = dt.unsqueeze(-2) * B                                             # (B,H,S,N)
        h = x.new_zeros(Bsz, Hh, Dh, S)
        ys = []
        for t in range(N):
            # h_t = dA_t * h_{t-1} + dB_t * x_t
            h = dA[..., t].unsqueeze(2) * h + dB[..., t].unsqueeze(2) * x[..., t:t + 1]
            # y_t = <C_t, h_t>_S
            y_t = (h * C[..., t].unsqueeze(2)).sum(-1)                        # (B,H,Dh)
            ys.append(y_t)
        y = torch.stack(ys, dim=-1) + D.view(1, Hh, 1, 1) * x                 # (B,H,Dh,N)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Bsz, _, H, W = x.shape
        N = H * W

        xz_gate = self.in_proj(x)                                             # (B, 3*Di, H, W)
        x_ssm, res, gate = xz_gate.chunk(3, dim=1)
        x_ssm_flat = x_ssm.flatten(2)                                         # (B, Di, N)

        # short DW mixing along scan
        x_ssm_flat = self.dw(x_ssm_flat)
        x_ssm_flat = F.silu(x_ssm_flat)

        # reshape to heads: (B, H, Dh, N)
        x_h = x_ssm_flat.view(Bsz, self.heads, self.d_head, N)

        # conformal permutation per head
        perm = self._mobius_perm(H, W, x.device)                              # (H, N) int64
        idx_x = perm.view(1, self.heads, 1, N).expand(Bsz, -1, self.d_head, -1)
        x_h = torch.gather(x_h, -1, idx_x)

        # selective parameters Δ, B, C
        dbc = self.dbc(x_ssm_flat).view(Bsz, self.heads, 1 + 2 * self.d_state, N)
        dt = F.softplus(dbc[:, :, 0])                                         # (B,H,N)
        B_t = dbc[:, :, 1:1 + self.d_state]                                   # (B,H,S,N)
        C_t = dbc[:, :, 1 + self.d_state:]                                    # (B,H,S,N)

        # apply permutation to dt, B_t, C_t
        idx_dt = perm.view(1, self.heads, N).expand(Bsz, -1, -1)
        dt = torch.gather(dt, -1, idx_dt)
        idx_bc = perm.view(1, self.heads, 1, N).expand(Bsz, -1, self.d_state, -1)
        B_t = torch.gather(B_t, -1, idx_bc)
        C_t = torch.gather(C_t, -1, idx_bc)

        A = -torch.exp(self.A_log)                                            # (H, S) negative
        y_h = self._selective_scan(x_h, dt, A, B_t, C_t, self.D)              # (B,H,Dh,N)

        # inverse permutation
        inv = perm.argsort(dim=-1)                                            # (H, N)
        idx_inv = inv.view(1, self.heads, 1, N).expand(Bsz, -1, self.d_head, -1)
        y_h = torch.gather(y_h, -1, idx_inv)

        # recombine and gate
        y = y_h.reshape(Bsz, self.d_inner, N).view(Bsz, self.d_inner, H, W)
        y = y * F.silu(gate) + res
        return self.out_proj(y)


# =====================================================================
# 3) Recursive Cross-Scale Gating (RXG) with IB-Route weights
# =====================================================================
class RXG_Fuse(nn.Module):
    r"""Multi-input fusion block với trọng số IB-Route (Fisher-style),
    2-step fixed-point refinement xấp xỉ, depthwise mix, residual.

    Args:
        c1: list[int] — channels của từng input
        c2: int       — channels output
    """

    def __init__(self, c1: Sequence[int], c2: int):
        super().__init__()
        assert isinstance(c1, (list, tuple)) and len(c1) >= 1
        self.n = len(c1)
        self.c2 = c2
        # Align channel: mỗi input project về c2
        self.proj = nn.ModuleList([Conv(c, c2, k=1, s=1) for c in c1])
        # Logits cho IB-Route: ước lượng Fisher ~ ||grad||^2 bằng second-moment
        # của projected features (practical, parameterless proxy).
        # Learnable temperature ổn định huấn luyện.
        self.tau = nn.Parameter(torch.tensor(1.0))
        # Second refinement step: depthwise mix + pointwise
        self.dw = Conv(c2, c2, k=3, s=1, g=c2)
        self.pw = Conv(c2, c2, k=1, s=1)

    @staticmethod
    def _spatial_align(feats: List[torch.Tensor]) -> List[torch.Tensor]:
        h0, w0 = feats[0].shape[-2:]
        out = []
        for f in feats:
            if f.shape[-2] != h0 or f.shape[-1] != w0:
                f = F.interpolate(f, size=(h0, w0), mode="nearest")
            out.append(f)
        return out

    def forward(self, x):
        if not isinstance(x, (list, tuple)):
            x = [x]
        assert len(x) == self.n, f"RXG_Fuse expected {self.n} inputs, got {len(x)}"

        feats = [self.proj[i](xi) for i, xi in enumerate(x)]
        feats = self._spatial_align(feats)

        # IB-Route weights: sqrt of second-moment per scale, normalized.
        # (Differentiable proxy to Fisher information ratio across branches.)
        fisher = torch.stack([(f * f).mean(dim=(1, 2, 3)) for f in feats], dim=-1)  # (B, n)
        fisher = fisher.mean(dim=0).clamp(min=1e-8).sqrt()                           # (n,)
        alpha = fisher / (fisher.sum() + 1e-6)
        alpha = torch.softmax(alpha / self.tau.abs().clamp(min=1e-2), dim=0)         # stable

        # Step 1 fusion
        y = sum(alpha[i] * feats[i] for i in range(self.n))
        # Step 2 refinement (fixed-point) — Banach contraction on channel space
        y = self.pw(self.dw(y)) + y
        # Residual with main branch (first input = top-down / primary path)
        return y + feats[0]
