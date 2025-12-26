# import torch
# import torch.nn as nn
#
# class BasicConv(nn.Module):  # https://arxiv.org/pdf/2010.03045.pdf
#     def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
#                  bn=True, bias=False):
#         super(BasicConv, self).__init__()
#         self.out_channels = out_planes
#         self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
#                               dilation=dilation, groups=groups, bias=bias)
#         self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
#         self.relu = nn.ReLU() if relu else None
#
#     def forward(self, x):
#         x = self.conv(x)
#         if self.bn is not None:
#             x = self.bn(x)
#         if self.relu is not None:
#             x = self.relu(x)
#         return x
#
#
# class ZPool(nn.Module):
#     def forward(self, x):
#         return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
#
#
# class AttentionGate(nn.Module):
#     def __init__(self):
#         super(AttentionGate, self).__init__()
#         kernel_size = 7
#         self.compress = ZPool()
#         self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)
#
#     def forward(self, x):
#         x_compress = self.compress(x)
#         x_out = self.conv(x_compress)
#         scale = torch.sigmoid_(x_out)
#         return x * scale
#
#
# class TripletAttention(nn.Module):
#     def __init__(self, no_spatial=False):
#         super(TripletAttention, self).__init__()
#         self.cw = AttentionGate()
#         self.hc = AttentionGate()
#         self.no_spatial = no_spatial
#         if not no_spatial:
#             self.hw = AttentionGate()
#
#     def forward(self, x):
#         x_perm1 = x.permute(0, 2, 1, 3).contiguous()
#         x_out1 = self.cw(x_perm1)
#         x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
#         x_perm2 = x.permute(0, 3, 2, 1).contiguous()
#         x_out2 = self.hc(x_perm2)
#         x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
#         if not self.no_spatial:
#             x_out = self.hw(x)
#             x_out = 1 / 3 * (x_out + x_out11 + x_out21)
#         else:
#             x_out = 1 / 2 * (x_out11 + x_out21)
#         return x_out
#


import torch
import torch.nn as nn


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ZPool(nn.Module):
    def forward(self, x):
        # Tối ưu: keepdim=True để tránh unsqueeze
        return torch.cat((torch.max(x, 1, keepdim=True)[0], torch.mean(x, 1, keepdim=True)), dim=1)


class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        # Chỉ trả về Attention Map (Weight), KHÔNG nhân với x ở đây
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid(x_out)
        return scale


class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()  # Channel-Width interaction
        self.hc = AttentionGate()  # Height-Channel interaction
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate()  # Height-Width interaction (Standard Spatial Attention)

    def forward(self, x):
        # Input: (N, C, H, W)

        # Branch 1: Channel-Width (Permute H to dim 1) -> Weights shape: (N, 1, C, W)
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        att1 = self.cw(x_perm1)
        att1 = att1.permute(0, 2, 1, 3)  # Permute mask back to (N, C, 1, W)

        # Branch 2: Height-Channel (Permute W to dim 1) -> Weights shape: (N, 1, H, C)
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        att2 = self.hc(x_perm2)
        att2 = att2.permute(0, 3, 2, 1)  # Permute mask back to (N, C, H, 1)

        # Branch 3: Spatial (No permute) -> Weights shape: (N, 1, H, W)
        if not self.no_spatial:
            att3 = self.hw(x)
            # Cộng broadcast các weights lại
            att = (att1 + att2 + att3) / 3
        else:
            att = (att1 + att2) / 2

        # Chỉ nhân 1 lần duy nhất tại đây
        return x * att