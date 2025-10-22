import torch
import torch.nn as nn
import math
import torch.nn.functional as F
class Position_embed(nn.Module):
    def __init__(self,max_L = 50,feature_num = 640):
        super(Position_embed, self).__init__()

        self.W = torch.nn.Parameter(torch.randn(max_L, feature_num) * 0.02)
    def forward(self,x):
        return x + self.W[:x.size(1)].unsqueeze(0)


class Encoder(nn.Module):
    def __init__(self, embed_dim=640, D=640, H=8, expansion_ratio=4):
        super(Encoder, self).__init__()
        self.embed_dim = embed_dim
        self.D = D
        self.H = H
        self.d = D // H

        # 使用Linear层而不是Parameter，这样更标准
        self.W_q = nn.Linear(embed_dim, D, bias=False)
        self.W_k = nn.Linear(embed_dim, D, bias=False)
        self.W_v = nn.Linear(embed_dim, D, bias=False)
        self.W_out = nn.Linear(D, embed_dim, bias=False)

        self.norm_attn = nn.LayerNorm(embed_dim)
        self.norm_mlp = nn.LayerNorm(embed_dim)

        self.linear1 = nn.Linear(embed_dim, embed_dim * expansion_ratio)
        self.linear2 = nn.Linear(embed_dim * expansion_ratio, embed_dim)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape

        residual = x

        # QKV投影
        q_proj = self.W_q(x)  # [B, L, D]
        k_proj = self.W_k(x)  # [B, L, D]
        v_proj = self.W_v(x)  # [B, L, D]

        # 多头分割 - 修正维度推断问题
        q = q_proj.view(batch_size, seq_len, self.H, self.d).transpose(1, 2)  # [B, H, L, d]
        k = k_proj.view(batch_size, seq_len, self.H, self.d).transpose(1, 2)  # [B, H, L, d]
        v = v_proj.view(batch_size, seq_len, self.H, self.d).transpose(1, 2)  # [B, H, L, d]

        # 注意力计算
        attn_scores = q @ k.transpose(-2, -1) / math.sqrt(self.d)  # [B, H, L, L]
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = attn_weights @ v  # [B, H, L, d]
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.D)  # [B, L, D]
        out = self.W_out(out)
        out = self.dropout(out)

        # 残差连接和层归一化
        out = residual + out
        out = self.norm_attn(out)

        # MLP部分
        residual_mlp = out
        out = self.linear1(out)
        out = F.gelu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.dropout(out)
        out = residual_mlp + out
        out = self.norm_mlp(out)

        return out

class Vit(nn.Module):
    def __init__(self,patch_size=8,embed_dim=640,num_heads=8,max_seq_length = 70,
                 encoder_num = 12,dropout = 0.3,num_classes = 10):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.p = 0

        self.embed = nn.Linear(patch_size * patch_size * 3, embed_dim)
        self.embed_dropout = nn.Dropout(dropout)
        self.cls_token = nn.Parameter(torch.rand(1,1, embed_dim))
        self.position_embedding = Position_embed(feature_num = embed_dim,max_L=max_seq_length)



        self.encoder_num = encoder_num
        self.encoders = nn.ModuleList(
            [
                Encoder(embed_dim=embed_dim,D=embed_dim) for _ in range(encoder_num)
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, image):
        # image(batch_size,3,32,32)
        batch_size, C, H, W = image.shape
        p = H//self.patch_size
        L = p*p
        patches = image.unfold(2, self.patch_size, self.patch_size) \
            .unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(batch_size, C, L, -1)
        patches = patches.permute(0, 2, 1, 3).contiguous().view(batch_size, L, -1)
        # [batch_size, L, patch_size * patch_size * 3]
        seq = self.embed(patches)
        seq = F.gelu(seq)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        seq = torch.cat([cls_tokens, seq], dim=1)
        seq = self.position_embedding(seq)
        seq = self.embed_dropout(seq)
        for i in range(self.encoder_num):
            seq = self.encoders[i](seq)
        # 分类准备
        cls_output = seq[:, 0]
        out = self.head(cls_output)
        return out

    def predict(self, image):
        with torch.no_grad():
            out = self.forward(image)
            out_p = F.softmax(out,dim=1)
        return torch.argmax(out_p, dim=1),out_p
