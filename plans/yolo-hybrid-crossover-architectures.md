# YOLO Cross-Architecture Hybrid Variants
## Combining Custom Architectures for Complementary Strengths

---

## 1. INVENTORY OF EXISTING ARCHITECTURES

| # | Architecture | Core Conv Innovation | Gate/Attention Innovation | Decomposition Axis |
|---|---|---|---|---|
| 1 | **Phoenix** | HeteroConv - DWConv 3x3 + DWConv 5x5 split | DualPoolGate - Avg+Max pool | Scale - 2 scales |
| 2 | **Chimera** | TridentConv - 3 dilations d=1,2,3 | SpectralGate - Mean+StdDev | Scale via dilation |
| 3 | **Nexus** | OmniDirConv - 4 directional kernels | NormRatioGate - L1/Linf sparsity | Direction - 4 dirs |
| 4 | **Prism V1** | DualFreqConv - LF AvgPool + HF residual | MCG - Mean Concentration Gate | Frequency - 2 bands |
| 5 | **Prism V2** | TriFreqConv - LF + MF + HF bands | FreqContrastGate + AdaptiveFreqRefine | Frequency - 3 bands |
| 6 | **Edge** | PConv - Partial Conv on 1/4 channels | None - lightweight focus | Sparsity - channel partial |
| 7 | **Spectra** | WaveletConv - Haar LL/LH/HL/HH subbands | DirectionalFreqGate | Frequency + Direction |
| 8 | **Zenith** | WaveletConv - learnable wavelet kernels | TopologicalGate - Euler characteristic | Wavelet + Topology |

---

## 2. COMPATIBILITY MATRIX

Key principle: **Combine architectures that decompose along ORTHOGONAL axes** for maximum complementarity.

```
             Phoenix  Chimera  Nexus  Prism  Edge  Spectra  Zenith
Phoenix        -       LOW     HIGH   HIGH   MED    MED     MED
Chimera       LOW       -      HIGH   HIGH   MED    MED     MED
Nexus         HIGH    HIGH       -     HIGH   MED    LOW     LOW
Prism         HIGH    HIGH     HIGH     -     HIGH   LOW     LOW
Edge          MED     MED      MED    HIGH     -    HIGH    HIGH
Spectra       MED     MED      LOW    LOW    HIGH     -     LOW
Zenith        MED     MED      LOW    LOW    HIGH   LOW      -
```

**HIGH** = orthogonal axes, strong synergy
**MED** = partial overlap, moderate synergy  
**LOW** = overlapping axes, redundant

---

## 3. TOP 5 HYBRID VARIANTS - Ranked by Novelty and Complementarity

### Hybrid 1: YOLO-Nexus-Prism — Direction + Frequency Fusion
**Complementarity: ★★★★★**

**Why it works:** Nexus decomposes spatially by direction, Prism decomposes spectrally by frequency. Together they cover the full frequency-direction feature space — similar to what Spectra does with wavelets, but using learned convolutions instead of fixed Haar basis.

```
NexusPrismBottleneck:
  Input → 1x1 Expand
       → Split into 2 halves
          ├── Half 1: OmniDirConv - 4 directional kernels - captures spatial structure
          └── Half 2: TriFreqConv - 3 frequency bands - captures spectral content
       → Concat → Channel Shuffle
       → NexusPrismGate - combines NormRatio + FreqContrast
       → 1x1 Project → Residual

Backbone: C3k2_NexusPrism
Neck:     NexusPrismCSP - uses same bottleneck
```

### Hybrid 2: YOLO-Prism-Edge — Frequency Intelligence + Lightweight
**Complementarity: ★★★★★**

**Why it works:** Edge processes only 1/4 channels with PConv for speed. Prism adds frequency decomposition. The hybrid applies frequency decomposition to the partial channels — getting both frequency awareness AND computational savings.

```
PrismEdgeBottleneck:
  Input → Split 1/4 vs 3/4
       ├── Active 1/4: TriFreqConv - frequency-decomposed processing
       └── Passive 3/4: Identity - zero compute
       → Concat → BN → SiLU
       → FreqContrastGate on full channels
       → 1x1 Project → Residual

Backbone: C3k2_PrismEdge
Neck:     Standard C2PSA or VoVGSCSP
```

### Hybrid 3: YOLO-Phoenix-Nexus — Multi-Scale + Multi-Direction
**Complementarity: ★★★★☆**

**Why it works:** Phoenix captures 2 scales (3x3 + 5x5), Nexus captures 4 directions. Together they provide scale AND directional diversity that neither achieves alone.

```
PhoenixNexusBottleneck:
  Input → 1x1 Expand
       → Split into 4 groups
          ├── G1: DWConv 3x3 - fine isotropic - from Phoenix
          ├── G2: DWConv 5x5 - coarse isotropic - from Phoenix  
          ├── G3: DWConv 1xK - horizontal strip - from Nexus
          └── G4: DWConv Kx1 - vertical strip - from Nexus
       → Concat → Channel Shuffle
       → DualPoolGate - from Phoenix
       → 1x1 Project → Residual

Backbone: C3k2_PhoenixNexus
Neck:     PhoenixNexusCSP
```

### Hybrid 4: YOLO-Chimera-Prism — Dilation-Scale + Frequency
**Complementarity: ★★★★☆**

**Why it works:** Chimera uses multi-dilation for scale, Prism uses frequency splitting. They decompose along orthogonal axes. Dilation captures multi-scale spatial context while frequency splitting captures spectral content.

```
ChimeraPrismBottleneck:
  Input → 1x1 Expand
       → Split into 2 halves
          ├── Half 1: TridentConv d=1,2,3 - multi-scale spatial
          └── Half 2: DualFreqConv LF+HF - spectral decomposition
       → Concat → Channel Shuffle
       → SpectralGate + MCG fusion
       → 1x1 Project → Residual

Backbone: C3k2_ChimeraPrism
Neck:     ChimeraPrismCSP
```

### Hybrid 5: YOLO-Spectra-Edge — Wavelet Full-Spectrum + Lightweight
**Complementarity: ★★★★☆**

**Why it works:** Spectra provides the richest feature decomposition (4 wavelet subbands) but is heavy. Edge's PConv principle makes it lightweight. Apply wavelet decomposition to only 1/4 active channels.

```
SpectraEdgeBottleneck:
  Input → Split 1/4 vs 3/4
       ├── Active 1/4: WaveletConv LL/LH/HL/HH decomposition
       └── Passive 3/4: Identity
       → Concat → BN → SiLU
       → DirectionalFreqGate
       → 1x1 Project → Residual

Backbone: C3k2_SpectraEdge
Neck:     VoVGSCSP from Edge
```

---

## 4. COMPARISON TABLE

| Hybrid | Backbone Axes | Expected Params | Expected GFLOPs | Innovation Level |
|---|---|---|---|---|
| Nexus-Prism | Direction + Frequency | Medium | Medium | Very High - two orthogonal axes |
| Prism-Edge | Frequency + Sparsity | Low | Low | High - freq-aware lightweight |
| Phoenix-Nexus | Scale + Direction | Medium | Medium | High - 4-path diversity |
| Chimera-Prism | Dilation + Frequency | Medium-High | Medium | High - spatial+spectral |
| Spectra-Edge | Wavelet + Sparsity | Low | Low | Medium - wavelet lightweight |

---

## 5. IMPLEMENTATION PLAN

For each hybrid:
1. Create `{name}_blocks.py` in `ultralytics/nn/modules/`
2. Register modules in `__init__.py` and `tasks.py`
3. Create YAML config in `ultralytics/cfg/models/11/`
4. Create test script `test_yolo_{name}.py`
5. Validate model builds and forward pass works

---

## 6. RECOMMENDED IMPLEMENTATION ORDER

1. **YOLO-Nexus-Prism** — highest complementarity, most novel
2. **YOLO-Prism-Edge** — practical lightweight variant
3. **YOLO-Phoenix-Nexus** — intuitive scale+direction combo
4. **YOLO-Chimera-Prism** — dilation+frequency combo
5. **YOLO-Spectra-Edge** — wavelet lightweight variant
