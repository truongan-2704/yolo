import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv


class PartialStarFusionBottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, star_ratio=0.5):
        """Initialize PartialStarFusionBottleneck with all V2 optimizations."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels

        # --- Channel reduction ---
        self.cv_reduce = Conv(c1, c_, 1, 1)

        # --- Partial channel split ---
        self.c_star = int(c_ * star_ratio)  # channels for star path
        self.c_bypass = c_ - self.c_star  # channels for bypass

        # === Shared DWConv base (applied to c_star channels) ===
        self.shared_dw = nn.Sequential(
            nn.Conv2d(self.c_star, self.c_star, kernel_size=3, padding=1,
                      groups=self.c_star, bias=False),
            nn.BatchNorm2d(self.c_star),
            nn.SiLU(),
        )

        # === Branch Local: learnable channel-wise scale on shared base ===
        self.local_scale = nn.Parameter(torch.ones(1, self.c_star, 1, 1))

        # === Branch Context: dilated + large-kernel on shared base ===
        self.deploy = False  # Flag đánh dấu trạng thái Deploy

        # Sub-branch 1: DWConv 3×3 dilation=2 (RF=7×7)
        self.ctx_dilated = nn.Sequential(
            nn.Conv2d(self.c_star, self.c_star, kernel_size=3, padding=2,
                      dilation=2, groups=self.c_star, bias=False),
            nn.BatchNorm2d(self.c_star),
        )

        # Sub-branch 2: DWConv 5×5 (RF=5×5)
        self.ctx_large = nn.Sequential(
            nn.Conv2d(self.c_star, self.c_star, kernel_size=5, padding=2,
                      groups=self.c_star, bias=False),
            nn.BatchNorm2d(self.c_star),
        )

        self.ctx_act = nn.SiLU()

        # === Star Operation Stabilizer ===
        self.star_bn = nn.BatchNorm2d(self.c_star)

        # === Channel expansion: c_ → c2 ===
        self.cv_expand = Conv(c_, c2, 1, 1)

        self.add = shortcut and c1 == c2

    @staticmethod
    def _channel_shuffle(x, groups=2):
        """Channel shuffle operation for cross-group information flow."""
        b, c, h, w = x.shape
        x = x.view(b, groups, c // groups, h, w)
        x = x.transpose(1, 2).contiguous()
        return x.view(b, c, h, w)

    @staticmethod
    def _param_free_gate(x):
        """Parameter-free spatial-channel joint gate (SimAM-inspired)."""
        mu = x.mean(dim=[2, 3], keepdim=True)  # [B, C, 1, 1]

        # TỐI ƯU 1: Tính phương sai thủ công, khắc phục hoàn toàn lỗi "degrees of freedom <= 0"
        var = (x - mu).pow(2).mean(dim=[2, 3], keepdim=True) + 1e-5  # [B, C, 1, 1]

        energy = (x - mu).pow(2) / var
        gate = torch.sigmoid(-energy + 1.0)
        return x * gate

    def forward(self, x):
        identity = x
        h = self.cv_reduce(x)
        h_star, h_bypass = h.split([self.c_star, self.c_bypass], dim=1)

        base = self.shared_dw(h_star)
        local_feat = base * self.local_scale

        # TỐI ƯU 2: Nhánh Context sẽ đi qua 1 lớp Conv duy nhất nếu đang ở chế độ deploy
        if self.deploy:
            ctx_feat = self.ctx_fused(base)
        else:
            ctx_feat = self.ctx_dilated(base) + self.ctx_large(base)

        ctx_feat = self.ctx_act(ctx_feat)

        star = local_feat * ctx_feat
        star = self.star_bn(star)
        star = self._param_free_gate(star)

        out = torch.cat([star, h_bypass], dim=1)
        out = self._channel_shuffle(out, groups=2)
        out = self.cv_expand(out)

        return out + identity if self.add else out

    def _fuse_conv_bn(self, conv, bn):
        """Hàm phụ trợ: Gộp trọng số của Conv và BatchNorm thành Conv có Bias."""
        w = conv.weight
        mean = bn.running_mean
        var_sqrt = torch.sqrt(bn.running_var + bn.eps)
        gamma = bn.weight
        beta = bn.bias

        # Tính toán trọng số và bias sau khi gộp
        w_fused = w * (gamma / var_sqrt).reshape(-1, 1, 1, 1)
        b_fused = -mean * gamma / var_sqrt + beta
        return w_fused, b_fused

    def switch_to_deploy(self):
        """
        Gộp nhánh ctx_dilated và ctx_large thành một lớp Conv 5x5 duy nhất.
        Được gọi tự động khi export model (ONNX, TensorRT).
        """
        if self.deploy:
            return

        # Gộp Conv+BN của nhánh 5x5
        w_large, b_large = self._fuse_conv_bn(self.ctx_large[0], self.ctx_large[1])

        # Gộp Conv+BN của nhánh 3x3 dilated
        w_dilated_3x3, b_dilated = self._fuse_conv_bn(self.ctx_dilated[0], self.ctx_dilated[1])

        # Padding trọng số của nhánh dilated (từ 3x3 dilation=2 lên 5x5 dilation=1)
        w_dilated_5x5 = torch.zeros_like(w_large)
        w_dilated_5x5[:, :, 0::2, 0::2] = w_dilated_3x3

        # Cộng dồn trọng số và bias của 2 nhánh
        w_fused = w_large + w_dilated_5x5
        b_fused = b_large + b_dilated

        # Khởi tạo lớp Conv mới để thay thế
        self.ctx_fused = nn.Conv2d(self.c_star, self.c_star, kernel_size=5, padding=2,
                                   groups=self.c_star, bias=True)
        self.ctx_fused.weight.data = w_fused
        self.ctx_fused.bias.data = b_fused

        # Đánh dấu trạng thái và xóa các nhánh thừa để giải phóng RAM
        self.deploy = True
        del self.ctx_large
        del self.ctx_dilated


class C3k2_DCNF_V2(nn.Module):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        self.c = int(c2 * e)

        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            PartialStarFusionBottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))