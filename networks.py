import time
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as pyd
from einops import repeat, rearrange
import numpy as np
import gin


class Normalization(nn.Module):
    def __init__(self, method: str, d_model: int):
        super().__init__()
        assert method in ["layer", "batch", "none"]
        if method == "layer":
            self.norm = nn.LayerNorm(d_model)
        elif method == "none":
            self.norm = lambda x: x
        else:
            self.norm = nn.BatchNorm1d(d_model)
        self.method = method

    def forward(self, x):
        if self.method == "batch":
            return self.norm(x.transpose(-1, 1)).transpose(-1, 1)
        return self.norm(x)


class FFBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.00, activation="gelu"):
        assert activation in ["gelu", "relu"]
        super().__init__()

        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(self, x):
        x1 = self.dropout(self.activation(self.ff1(x)))
        x1 = self.dropout(self.activation(self.ff2(x1)))
        return x + x1


@gin.configurable(denylist=["state_dim"])
class FeedForwardEncoder(nn.Module):
    def __init__(
        self,
        state_dim,
        d_model=128,
        d_ff=512,
        dropout=0.00,
        activation="gelu",
        norm="none",
    ):
        super().__init__()
        self.traj_emb = nn.Linear(state_dim, d_model)
        self.traj_block = FFBlock(d_model, d_ff, dropout=dropout, activation=activation)
        self.traj_last = nn.Linear(d_model, d_model)
        self.norm = Normalization(norm, d_model)

        self.activation = F.gelu if activation == "gelu" else F.relu
        self.dropout = nn.Dropout(dropout)
        self.emb_dim = d_model
        self.needs_mask = False

    def forward(self, states, mask=None):
        assert mask is None
        traj_emb = self.dropout(self.activation(self.traj_emb(states)))
        traj_emb = self.traj_block(traj_emb)
        traj_emb = self.traj_last(traj_emb)
        traj_emb = self.dropout(self.norm(traj_emb))
        return traj_emb


class FullAttention(nn.Module):
    def __init__(
        self,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, qkv, attn_mask, output_attn=False):
        queries, keys, values = torch.unbind(qkv, dim=2)
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = 1.0 / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)  # expand heads dimension
            scores.masked_fill_(attn_mask, -torch.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        return V, A


class FlashAttention(nn.Module):
    def __init__(self, attention_dropout: float = 0.0):
        try:
            from flash_attn.flash_attention import FlashAttention as _FlashAttention
        except ImportError:
            raise ImportError("Missing flash attention; pip install flash-attn")
        super().__init__()
        self.attn = _FlashAttention(attention_dropout=attention_dropout)

    def forward(self, qkv, attn_mask, output_attn=False):
        # flash-attention bert_padding.unpad_input says True is valid
        breakpoint()
        key_padding_mask = ~attn_mask[:, :, 0]
        output, attn = self.attn(qkv, key_padding_mask=key_padding_mask, causal=True)
        return output, attn


class AttentionLayer(nn.Module):
    def __init__(
        self,
        attention,
        d_model,
        d_qkv,
        n_heads,
        dropout_qkv=0.0,
    ):
        super().__init__()
        self.attention = attention
        self.qkv_projection = nn.Linear(d_model, 3 * d_qkv * n_heads, bias=False)
        self.dropout_qkv = nn.Dropout(dropout_qkv)
        self.out_projection = nn.Linear(d_qkv * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, sequence, attn_mask):
        qkv = self.dropout_qkv(self.qkv_projection(sequence))
        qkv = rearrange(
            qkv,
            "batch len (three d_qkv heads) -> batch len three heads d_qkv",
            heads=self.n_heads,
            three=3,
        )
        out, attn = self.attention(
            qkv=qkv,
            attn_mask=attn_mask,
        )
        out = rearrange(out, "batch len heads dim -> batch len (heads dim)")
        out = self.out_projection(out)
        return out, attn


class TransformerLayer(nn.Module):
    def __init__(
        self,
        self_attention,
        d_model,
        d_ff,
        dropout_ff=0.1,
        activation="gelu",
        norm="layer",
    ):
        super().__init__()
        self.self_attention = self_attention
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.norm1 = Normalization(method=norm, d_model=d_model)
        self.norm2 = Normalization(method=norm, d_model=d_model)
        self.norm3 = Normalization(method=norm, d_model=d_model)
        self.dropout_ff = nn.Dropout(dropout_ff)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(self, self_seq, self_mask=None):
        q1 = self.norm1(self_seq)
        q1, self_attn = self.self_attention(sequence=q1, attn_mask=self_mask)
        self_seq = self_seq + q1
        q1 = self.norm3(self_seq)
        q1 = self.dropout_ff(self.activation(self.ff1(q1)))
        q1 = self.dropout_ff(self.ff2(q1))
        self_seq = self_seq + q1
        return self_seq, {"self_attn": self_attn}


@gin.configurable(denylist=["state_dim"])
class TransformerEncoder(nn.Module):
    def __init__(
        self,
        state_dim: int,
        max_seq_len: int,
        d_model: int = 64,
        d_ff: int = 256,
        d_emb_ff: int = None,
        n_heads: int = 4,
        layers: int = 4,
        dropout_emb: float = 0.05,
        dropout_ff: float = 0.1,
        dropout_attn: float = 0.3,
        dropout_qkv: float = 0.05,
        activation: str = "gelu",
        attention: str = "vanilla",
        norm: str = "layer",
    ):
        super().__init__()
        assert activation in ["gelu", "relu"]
        assert attention in ["vanilla", "flash"]

        # embedding
        self.max_seq_len = max_seq_len
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim=d_model)
        d_emb_ff = d_emb_ff or d_model
        self.obs_embedding = nn.Sequential(
            nn.Linear(state_dim, d_emb_ff),
            nn.LeakyReLU(),
            nn.Linear(d_emb_ff, d_model),
            Normalization(method=norm, d_model=d_model),
        )

        self.activation = F.gelu if activation == "gelu" else F.relu
        self.dropout = nn.Dropout(dropout_emb)

        head_dim = d_model // n_heads
        if attention == "flash":
            assert head_dim in range(8, 129, 8)

        def make_attn():
            attnCls = FullAttention if attention == "vanilla" else FlashAttention
            return AttentionLayer(
                attention=attnCls(attention_dropout=dropout_attn),
                d_model=d_model,
                d_qkv=head_dim,
                n_heads=n_heads,
                dropout_qkv=dropout_qkv,
            )

        def make_layer():
            return TransformerLayer(
                self_attention=make_attn(),
                d_model=d_model,
                d_ff=d_ff,
                dropout_ff=dropout_ff,
                activation=activation,
                norm=norm,
            )

        self.layers = nn.ModuleList([make_layer() for _ in range(layers)])
        self.norm = Normalization(method=norm, d_model=d_model)
        self.d_model = d_model
        self.needs_mask = True

    @property
    def emb_dim(self):
        return self.d_model

    def forward(self, states, mask):
        assert mask is not None
        batch, length, dim = states.shape
        assert length <= self.max_seq_len
        pos_idxs = torch.arange(length, device=states.device, dtype=torch.long)
        pos_emb = self.position_embedding(pos_idxs)
        pos_emb = repeat(pos_emb, f"length d_model -> {batch} length d_model")
        traj_emb = self.obs_embedding(states)
        traj_emb = self.dropout(traj_emb + pos_emb)
        # self-attention
        for layer in self.layers:
            traj_emb, attn = layer(self_seq=traj_emb, self_mask=mask)
        traj_emb = self.norm(traj_emb)
        return traj_emb


class _Categorical(pyd.Categorical):
    def sample(self, *args, **kwargs):
        return super().sample(*args, **kwargs).unsqueeze(-1)


@gin.configurable(denylist=["state_dim", "action_dim"])
class Actor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_layers: int = 2,
        d_hidden: int = 256,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        self.in_layer = nn.Linear(state_dim, d_hidden)
        self.dropout = nn.Dropout(dropout_p)
        self.layers = nn.ModuleList(
            [nn.Linear(d_hidden, d_hidden) for _ in range(n_layers - 1)]
        )
        self.out_layer = nn.Linear(d_hidden, action_dim)

    def forward(self, state):
        x = self.dropout(F.leaky_relu(self.in_layer(state)))
        for layer in self.layers:
            x = self.dropout(F.leaky_relu(layer(x)))
        dist_params = self.out_layer(x)
        return _Categorical(logits=dist_params)
