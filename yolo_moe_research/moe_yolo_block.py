import torch
import torch.nn as nn
import torch.nn.functional as F

class LightExpert(nn.Module):
    """
    Expert 1: Light CNN (3x3 Depthwise Conv)
    Chuyên xử lý các vùng background, trời, mặt đường... rất tiết kiệm FLOPs.
    """
    def __init__(self, dim):
        super().__init__()
        # Depthwise convolution để tiết kiệm tối đa tham số
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class DenseExpert(nn.Module):
    """
    Expert 2: Dense CNN (7x7 RepConv-style)
    Chuyên xử lý biên vật thể, texture chi tiết. Receptive field lớn.
    """
    def __init__(self, dim):
        super().__init__()
        # Large kernel depthwise
        self.conv_large = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=False)
        self.bn1 = nn.BatchNorm2d(dim)
        self.act = nn.SiLU()
        # Pointwise để trộn channel
        self.pw_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        x = self.act(self.bn1(self.conv_large(x)))
        return self.bn2(self.pw_conv(x))

class TransformerExpert(nn.Module):
    """
    Expert 3: Transformer Block (Global/Local Attention)
    Chuyên xử lý các vật thể che khuất, cần ngữ nghĩa toàn cục.
    """
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(1, dim) # LayerNorm equivalent cho dạng NCHW
        self.act = nn.SiLU()

    def forward(self, x):
        B, C, H, W = x.shape
        x_norm = self.norm(x)
        
        # Tạo Q, K, V
        qkv = self.qkv(x_norm).reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2] # (B, heads, head_dim, N)
        
        # Attention: (B, heads, N, head_dim) x (B, heads, head_dim, N) -> (B, heads, N, N)
        # Cảnh báo: Nếu H*W quá lớn, phép toán này sẽ tốn memory. 
        # Trong thực tế, Expert này chỉ áp dụng cho độ phân giải thấp (Stage 4, 5).
        attn = (q.transpose(-2, -1) @ k) * ((C // self.num_heads) ** -0.5)
        attn = attn.softmax(dim=-1)
        
        # Output: (B, heads, N, N) x (B, heads, N, head_dim) -> (B, heads, head_dim, N)
        out = (attn @ v.transpose(-2, -1)).transpose(-2, -1)
        out = out.reshape(B, C, H, W)
        
        return x + self.act(self.proj(out))

class SpatialAwareDynamicModalityRouter(nn.Module):
    """
    Khối Core: Định tuyến Không gian Động (Spatial-Aware Dynamic Modality Routing)
    """
    def __init__(self, dim, num_experts=3):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        
        # Gating Network siêu nhẹ
        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim // 4),
            nn.SiLU(),
            nn.Conv2d(dim // 4, num_experts, kernel_size=1)
        )
        
        # Khởi tạo 3 Experts
        self.experts = nn.ModuleList([
            LightExpert(dim),
            DenseExpert(dim),
            TransformerExpert(dim)
        ])

    def forward(self, x):
        # 1. Dự đoán Gate Probabilities cho MỖI spatial pixel (H, W)
        # gate_logits shape: (B, 3, H, W)
        gate_logits = self.gate(x)
        
        # Tính xác suất qua Softmax ở chiều num_experts (dim=1)
        # routing_weights shape: (B, 3, H, W)
        routing_weights = F.softmax(gate_logits, dim=1)
        
        # 2. Xử lý qua các Experts
        # *Note*: Trong file này sử dụng Soft-Routing (Mix), dễ hội tụ khi train. 
        # Khi đẩy lên TensorRT/C++ inference, ta sẽ dùng Hard-Routing: 
        # mask = torch.argmax(routing_weights, dim=1) để kích hoạt Sparse Conv/Sparse Attention.
        
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            # Tính toán qua hàm Expert
            E_x = expert(x) 
            
            # Nhân với trọng số routing của Expert đó tại từng điểm pixel
            # routing_weights[:, i:i+1, :, :] -> shape (B, 1, H, W) 
            # E_x -> shape (B, C, H, W)
            weighted_E_x = E_x * routing_weights[:, i:i+1, :, :]
            expert_outputs.append(weighted_E_x)
            
        # 3. Gộp feature theo chiều tổng
        out = sum(expert_outputs)
        return out

class MoEYOLOBlock(nn.Module):
    """
    MoE-YOLO Bottleneck. Thay thế cho khối C2f/Bottleneck cơ bản của YOLO.
    """
    def __init__(self, c1, c2, shortcut=True):
        super().__init__()
        self.cv1 = nn.Conv2d(c1, c2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c2)
        self.act1 = nn.SiLU()
        
        # Thay vì 2 lớp Conv 3x3 như YOLO cũ, ta chèn khối MoE định tuyến động
        self.moe_block = SpatialAwareDynamicModalityRouter(dim=c2)
        self.shortcut = shortcut and c1 == c2

    def forward(self, x):
        identity = x
        x = self.act1(self.bn1(self.cv1(x)))
        out = self.moe_block(x)
        if self.shortcut:
            out = out + identity
        return out

if __name__ == "__main__":
    print("=== Khởi tạo MoE-YOLO: Spatial-Aware Dynamic Modality Routing ===")
    
    # Giả lập 1 Tensor Input từ Layer sâu (Stage 4) của YOLO (VD: stride 16)
    # Batch = 2, Channels = 256, H = 40, W = 40 (Ảnh 640x640)
    B, C, H, W = 2, 256, 40, 40
    dummy_input = torch.randn(B, C, H, W)
    
    print(f"-> Tensor đầu vào: {dummy_input.shape}")
    
    # Khởi tạo Block bảo toàn kênh
    block = MoEYOLOBlock(c1=256, c2=256, shortcut=True)
    
    # Forward Pass
    try:
        output = block(dummy_input)
        print(f"-> Tensor đầu ra: {output.shape}")
        print("=> SUCCESS: Khối MoE đã hoạt động và xử lý thành công qua 3 Experts!")
        
        # Test routing mask
        gate_out = block.moe_block.gate(dummy_input)
        gate_probs = F.softmax(gate_out, dim=1)
        print(f"-> Gate Probabilities Shape (B, 3 Experts, H, W): {gate_probs.shape}")
        
    except Exception as e:
        print(f"=> LỖI: {e}")

