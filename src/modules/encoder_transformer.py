import einops
import torch
import torch.nn.functional as F
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=64, n_head=4, bias=True, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias)
        self.out_proj = nn.Linear(d_model, d_model, bias)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        batch, seq_len, hidden_dim = x.shape
        qkv = self.qkv_proj(x)
        qkv = einops.rearrange(qkv, "b l (s h d) -> b s h l d", h=self.n_head, s=3)
        q, k, v = qkv.unbind(1)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p)
        y = y.permute(0, 2, 1, 3).flatten(2)
        return self.dropout(self.out_proj(y))


class MLP(nn.Module):
    def __init__(self, d_model=64, mlp_ratio=4, bias=True, dropout=0.1):
        super().__init__()
        self.up_proj = nn.Linear(d_model, mlp_ratio * d_model, bias)
        self.down_proj = nn.Linear(mlp_ratio * d_model, d_model, bias)
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.up_proj(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(self, d_model=64, mlp_ratio=4, n_head=4, bias=True, dropout=0.1):
        super().__init__()
        self.pre_norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_head, bias, dropout)
        self.pre_norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, mlp_ratio, bias, dropout)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        x = einops.rearrange(x, "b c h w -> b (h w) c")
        x = x + self.attn(self.pre_norm1(x))
        x = x + self.mlp(self.pre_norm2(x))
        x = einops.rearrange(x, "b (h w) c -> b c h w", h=height, w=width)
        return x


if __name__ == "__main__":
    rand = torch.randn(4, 64, 32, 32)
    tr = Transformer()
    tr(rand)
