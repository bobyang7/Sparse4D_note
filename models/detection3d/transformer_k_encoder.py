import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    # 这里的dropout应该设置为多少呢
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            # 这里的GELU函数是什么样的，有什么特点
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim, heads=heads, dim_head=dim_head, dropout=dropout
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class Transformer_k_encoder(nn.Module):
    def __init__(self, dim=256, depth=6, heads=8, dim_head=32, mlp_dim=1024, dropout=0):
        super().__init__()
        self.k = nn.Parameter(torch.randn(1, 256))
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(
        self,
        before_align_anchor_embed,
        after_align_anchor_embed,
        anchor_embed,
        cached_k_pred,
        ego_embed,
    ):
        input_embed = torch.cat(
            (
                before_align_anchor_embed[:, None],
                after_align_anchor_embed[:, None],
                anchor_embed[:, None],
                cached_k_pred[:, None],
                ego_embed[:, None],
            ),
            dim=-2,
        )
        num_anchor = input_embed.shape[0]
        k = torch.tile(self.k[None], (num_anchor, 1, 1))
        x = torch.cat((k, input_embed), dim=-2)
        x = self.transformer(x)
        return x
