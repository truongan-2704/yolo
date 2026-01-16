import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv
from torchvision.models.resnet import resnet50

__all__ = ["BiFPN_Concat", "BiFPN"]

def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class BiFPN_Concat(nn.Module):
    def __init__(self, c1, c2):
        super(BiFPN_Concat, self).__init__()
        self.w1_weight = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w2_weight = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        self.conv = Conv(c1, c2, 1, 1, 0)
        self.act = nn.ReLU()

    def forward(self, x):
        if len(x) == 2:
            w = self.w1_weight
            weight = w / (torch.sum(w, dim=0) + self.epsilon)
            x = self.conv(self.act(weight[0] * x[0] + weight[1] * x[1]))
        elif len(x) == 3:
            w = self.w2_weight
            weight = w / (torch.sum(w, dim=0) + self.epsilon)
            x = self.conv(self.act(weight[0] * x[0] + weight[1] * x[1] + weight[2] * x[2]))
        return x


# class swish(nn.Module):
#     def forward(self, x):
#         return x * torch.sigmoid(x)
#
#
# class BiFPN(nn.Module):
#     def __init__(self, length):
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(length, dtype=torch.float32), requires_grad=True)
#         self.swish = swish()
#         self.epsilon = 0.0001
#
#     def forward(self, x):
#         weights = self.weight / (torch.sum(self.swish(self.weight), dim=0) + self.epsilon)
#         weighted_feature_maps = [weights[i] * x[i] for i in range(len(x))]
#         stacked_feature_maps = torch.stack(weighted_feature_maps, dim=0)
#         result = torch.sum(stacked_feature_maps, dim=0)
#         return result
#
#
# import torch
# import torch.nn as nn

# class BiFPN(nn.Module):
#     def __init__(self, ch, c2):
#         # ch: list số kênh đầu vào (VD: [1024, 512])
#         # c2: số kênh đầu ra mong muốn (VD: 512)
#         super().__init__()
#
#         # Trọng số học được cho việc trộn đặc trưng
#         self.w = nn.Parameter(torch.ones(len(ch), dtype=torch.float32), requires_grad=True)
#         self.epsilon = 1e-4
#
#         # --- QUAN TRỌNG: Tạo Conv1x1 để ép tất cả đầu vào về cùng kích thước c2 ---
#         self.convs = nn.ModuleList()
#         for c_in in ch:
#             if c_in != c2:
#                 # Nếu kênh khác nhau -> Dùng Conv 1x1 để đổi kênh
#                 self.convs.append(nn.Conv2d(c_in, c2, 1, stride=1, padding=0))
#             else:
#                 # Nếu kênh giống nhau -> Giữ nguyên (Tiết kiệm tính toán)
#                 self.convs.append(nn.Identity())
#
#         self.act = nn.SiLU()  # Hoặc dùng Swish như bạn muốn
#
#     def forward(self, x):
#         # x là list các tensor [x1, x2, ...]
#         # 1. Chuẩn hóa trọng số (Fast normalized fusion)
#         w = self.w
#         weight = w / (torch.sum(w, dim=0) + self.epsilon)
#         # 2. Xử lý từng đầu vào: Đổi kênh (nếu cần) -> Nhân trọng số -> Cộng dồn
#         # Chúng ta cộng trực tiếp (sum) thay vì stack để tiết kiệm bộ nhớ và tránh lỗi
#         out = 0
#         for i in range(len(x)):
#             # self.convs[i](x[i]) đảm bảo tensor luôn có số kênh là c2
#             out += weight[i] * self.convs[i](x[i])
#
#         return self.act(out)

#
# class BiFPN(nn.Module):
#     def __init__(self, ch, c2):
#         # ch: list số kênh đầu vào
#         # c2: số kênh đầu ra mong muốn
#         super().__init__()
#
#         self.w = nn.Parameter(torch.ones(len(ch), dtype=torch.float32), requires_grad=True)
#         self.epsilon = 1e-4
#
#         # --- ACTIVE MODE: LUÔN DÙNG CONV ĐỂ HỌC ---
#         # Không dùng Identity nữa để đảm bảo mAP cao nhất
#         self.convs = nn.ModuleList()
#         for c_in in ch:
#             self.convs.append(nn.Conv2d(c_in, c2, 1, stride=1, padding=0))
#
#         self.act = nn.SiLU()
#
#     def forward(self, x):
#         # 1. Trọng số
#         w = self.w
#         weight = w / (torch.sum(w, dim=0) + self.epsilon)
#
#         # 2. Tính toán
#         out = 0
#         for i in range(len(x)):
#             # Luôn đi qua Conv để biến đổi đặc trưng trước khi cộng
#             out += weight[i] * self.convs[i](x[i])
#
#         return self.act(out)

class BiFPN(nn.Module):
    def __init__(self, ch, c2):
        super().__init__()

        self.w = nn.Parameter(torch.ones(len(ch), dtype=torch.float32), requires_grad=True)
        self.epsilon = 1e-4
        self.convs = nn.ModuleList()

        for c_in in ch:
            self.convs.append(Conv(c_in, c2, 1, 1))

        self.act = nn.SiLU()

    def forward(self, x):

        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)
        out = 0
        for i in range(len(x)):
            out += weight[i] * self.convs[i](x[i])
        return self.act(out)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from .conv import Conv  # Import Conv chuẩn từ Ultralytics
#
#
# class BiFPN(nn.Module):
#     def __init__(self, c1, c2):
#         super().__init__()
#         # Xử lý linh hoạt đầu vào c1 (có thể là int hoặc list)
#         if isinstance(c1, int):
#             c1 = [c1]
#
#         # Trọng số học được
#         self.w = nn.Parameter(torch.ones(len(c1), dtype=torch.float32), requires_grad=True)
#         self.epsilon = 1e-4
#
#         # Các lớp Conv để đồng bộ hóa Channels
#         self.convs = nn.ModuleList([Conv(x, c2, 1, 1) for x in c1])
#
#         self.act = nn.SiLU()
#
#     def forward(self, x):
#         # Trọng số chuẩn hóa
#         w = self.w
#         weight = w / (torch.sum(w, dim=0) + self.epsilon)
#
#         # Lấy kích thước mục tiêu từ input đầu tiên (thường là input tại tầng hiện tại trong YAML)
#         # Ví dụ: Tại tầng P3, input đầu tiên thường là P3 gốc -> target_size chuẩn.
#         target_size = x[0].shape[2:]
#
#         out = 0
#         for i in range(len(x)):
#             # 1. Đồng bộ Channel
#             feat = self.convs[i](x[i])
#
#             # 2. Đồng bộ Spatial Size (QUAN TRỌNG: Resize an toàn)
#             if feat.shape[2:] != target_size:
#                 feat = F.interpolate(feat, size=target_size, mode='nearest')
#
#             # 3. Cộng dồn
#             out += weight[i] * feat
#
#         return self.act(out)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.mhsa = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, -1).permute(0, 2, 1)  # (B, Seq_len, Dim)
        attn_output, _ = self.mhsa(x, x, x)
        attn_output = self.norm(attn_output + x)  # Residual Connection
        return attn_output.permute(0, 2, 1).view(b, c, h, w)  # Chuyển về lại


class BiFPN_Transformer(nn.Module):
    def __init__(self, length, embed_dim=128, num_heads=4):
        super().__init__()
        self.length = length
        self.weight = nn.Parameter(torch.ones(length, dtype=torch.float32), requires_grad=True)
        self.epsilon = 1e-4  # Giá trị epsilon nhỏ hơn
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)

    def forward(self, x):
        device = x[0].device  # Lấy thiết bị của tensor đầu vào
        weights = self.weight.to(device)  # Chuyển weight lên đúng thiết bị
        norm_weights = weights / (torch.sum(F.silu(weights), dim=0) + self.epsilon)

        weighted_feature_maps = [norm_weights[i] * x[i] for i in range(self.length)]
        stacked_feature_maps = torch.stack(weighted_feature_maps, dim=0)
        result = torch.sum(stacked_feature_maps, dim=0)

        result = self.attention(result)  # Áp dụng Multi-Head Self-Attention
        return result
