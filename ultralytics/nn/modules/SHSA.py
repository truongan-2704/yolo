import torch
import torch.nn as nn
import torch.nn.functional as F
class SHSA(nn.Module):
    """
    Spatial-Head Self-Attention: attention theo không gian (HW tokens), chia head theo channel.
    """
    def __init__(self, channels, heads=4, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert channels % heads == 0, "channels must be divisible by heads"
        self.channels = channels
        self.heads = heads
        self.dim_head = channels // heads

        self.to_q = nn.Conv2d(channels, channels, 1, bias=qkv_bias)
        self.to_k = nn.Conv2d(channels, channels, 1, bias=qkv_bias)
        self.to_v = nn.Conv2d(channels, channels, 1, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(channels, channels, 1, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        b, c, h, w = x.shape
        n = h * w

        q = self.to_q(x).view(b, self.heads, self.dim_head, n)  # (B, H, Dh, N)
        k = self.to_k(x).view(b, self.heads, self.dim_head, n)
        v = self.to_v(x).view(b, self.heads, self.dim_head, n)

        # transpose to (B, H, N, Dh) for q; (B, H, Dh, N) for k/v as SDPA expects
        q = q.permute(0, 1, 3, 2).contiguous()  # (B,H,N,Dh)
        k = k                                      # (B,H,Dh,N)
        v = v

        # scaled dot-product attention over spatial tokens
        # F.scaled_dot_product_attention expects (..., L, E) shapes for q,k,v
        # Here, treat H heads as batch dims via view
        q_ = q.reshape(b * self.heads, n, self.dim_head)         # (B*H, N, Dh)
        k_ = k.reshape(b * self.heads, self.dim_head, n)         # (B*H, Dh, N)
        v_ = v.reshape(b * self.heads, self.dim_head, n)         # (B*H, Dh, N)

        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q_, k_.transpose(1, 2), v_.transpose(1, 2)  # (B*H,N,Dh),(B*H,N,Dh),(B*H,N,Dh)
        )  # -> (B*H, N, Dh)

        attn_out = self.attn_drop(attn_out)

        # reshape back
        attn_out = attn_out.view(b, self.heads, n, self.dim_head).permute(0, 1, 3, 2).contiguous()
        attn_out = attn_out.view(b, self.channels, n).view(b, self.channels, h, w)

        out = self.proj(attn_out)
        out = self.proj_drop(out)
        return out
