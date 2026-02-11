import torch
import torch.nn as nn

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class SE(nn.Module):
    def __init__(self, c, r=4):
        super().__init__()
        c_reduced = max(1, c // r)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c, c_reduced, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(c_reduced, c, 1, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(self.pool(x))

class FusedMBConv(nn.Module):
    def __init__(self, c1, c2, n=1, s=1, expand=4):
        super().__init__()
        hidden = int(c1 * expand)
        self.add = s == 1 and c1 == c2

        self.block = nn.Sequential(
            # Sử dụng autopad(3) để đảm bảo đầu ra đúng kích thước khi s=2
            nn.Conv2d(c1, hidden, 3, s, autopad(3), bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            # Pointwise
            nn.Conv2d(hidden, c2, 1, 1, autopad(1), bias=False),
            nn.BatchNorm2d(c2),
        )

    def forward(self, x):
        return x + self.block(x) if self.add else self.block(x)

class MBConv(nn.Module):
    def __init__(self, c1, c2, n=1, s=1, expand=4):
        super().__init__()
        hidden = int(c1 * expand)
        self.add = s == 1 and c1 == c2

        self.block = nn.Sequential(
            # Expansion
            nn.Conv2d(c1, hidden, 1, 1, autopad(1), bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            # Depthwise
            nn.Conv2d(hidden, hidden, 3, s, autopad(3), groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            # Squeeze-and-Excitation
            SE(hidden),
            # Pointwise
            nn.Conv2d(hidden, c2, 1, 1, autopad(1), bias=False),
            nn.BatchNorm2d(c2),
        )

    def forward(self, x):
        return x + self.block(x) if self.add else self.block(x)