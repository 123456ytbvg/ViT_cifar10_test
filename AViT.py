import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class AdaptivePositionEmbedding(nn.Module):
    """自适应位置编码，可学习且更轻量"""

    def __init__(self, max_seq_length=70, embed_dim=640):
        super(AdaptivePositionEmbedding, self).__init__()
        self.position_embedding = nn.Parameter(torch.randn(1, max_seq_length, embed_dim) * 0.02)

    def forward(self, x):
        return x + self.position_embedding[:, :x.size(1)]


class AdaptiveAttention(nn.Module):
    """自适应注意力机制，减少计算复杂度"""

    def __init__(self, embed_dim=640, num_heads=8, reduction_ratio=4):
        super(AdaptiveAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.reduction_ratio = reduction_ratio

        # 线性投影
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)

        # 自适应token选择 - 修复维度问题
        if reduction_ratio > 1:
            self.token_reduction = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // reduction_ratio),
                nn.GELU(),
                nn.Linear(embed_dim // reduction_ratio, embed_dim)
            )

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B, N, C = x.shape

        # 自适应token缩减（可选）- 修复实现
        if hasattr(self, 'token_reduction') and N > 10 and self.reduction_ratio > 1:
            # 使用平均池化进行token缩减
            if N % self.reduction_ratio == 0:
                x_adapted = x.reshape(B, N // self.reduction_ratio, self.reduction_ratio, C).mean(dim=2)
            else:
                # 如果无法整除，使用线性投影
                x_adapted = self.token_reduction(x)
        else:
            x_adapted = x

        # QKV投影
        qkv = self.qkv(x_adapted).reshape(B, -1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 缩放点积注意力
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # 注意力输出
        x_attn = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        x_attn = self.proj(x_attn)
        x_attn = self.dropout(x_attn)

        return x_attn


class EfficientFFN(nn.Module):
    """高效的前馈网络"""

    def __init__(self, embed_dim=640, expansion_ratio=3):
        super(EfficientFFN, self).__init__()
        hidden_dim = int(embed_dim * expansion_ratio)

        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        return self.net(x)


class AViTBlock(nn.Module):
    """AViT基础块，结合自适应注意力和高效FFN"""

    def __init__(self, embed_dim=640, num_heads=8, reduction_ratio=4, ffn_expansion=3):
        super(AViTBlock, self).__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.adaptive_attention = AdaptiveAttention(embed_dim, num_heads, reduction_ratio)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.efficient_ffn = EfficientFFN(embed_dim, ffn_expansion)

    def forward(self, x):
        # 自适应注意力 + 残差
        x = x + self.adaptive_attention(self.norm1(x))
        # 高效FFN + 残差
        x = x + self.efficient_ffn(self.norm2(x))
        return x


class PatchEmbedding(nn.Module):
    """改进的patch嵌入，使用卷积代替线性层"""

    def __init__(self, patch_size=8, embed_dim=640, in_channels=3):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # [B, C, H, W] -> [B, E, H/P, W/P] -> [B, E, N] -> [B, N, E]
        x = self.proj(x)
        B, E, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class AViT(nn.Module):
    """轻量化的Adaptive Vision Transformer"""

    def __init__(self,
                 patch_size=8,
                 embed_dim=512,  # 减少嵌入维度
                 num_heads=8,
                 max_seq_length=70,
                 encoder_num=8,  # 减少层数
                 reduction_ratio=2,  # token缩减比例
                 dropout=0.1,  # 减少dropout
                 num_classes=10):
        super().__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbedding(patch_size, embed_dim)

        # 计算序列长度
        self.seq_length = (32 // patch_size) ** 2 + 1  # +1 for cls token

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = AdaptivePositionEmbedding(max_seq_length, embed_dim)
        self.embed_dropout = nn.Dropout(dropout)

        # 使用AViT块
        self.encoder_num = encoder_num
        self.encoders = nn.ModuleList([
            AViTBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                reduction_ratio=reduction_ratio,
                ffn_expansion=2  # 减少FFN扩展比例
            ) for _ in range(encoder_num)
        ])

        # 简化的分类头
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Parameter):
            torch.nn.init.trunc_normal_(m, std=0.02)

    def forward(self, x):
        batch_size = x.shape[0]

        # Patch嵌入
        x = self.patch_embed(x)  # [B, N, E]

        # 添加CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # 位置编码和dropout
        x = self.pos_embed(x)
        x = self.embed_dropout(x)

        # 通过编码器
        for encoder in self.encoders:
            x = encoder(x)

        # 分类
        cls_output = x[:, 0]
        out = self.head(cls_output)
        return out

    def predict(self, x):
        with torch.no_grad():
            out = self.forward(x)
            out_p = F.softmax(out, dim=1)
        return torch.argmax(out_p, dim=1), out_p