import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, Any

# torch.rand(B, S, dtype=torch.long)  # B=batch_size, S=sequence_length
class MyModel(nn.Module):
    def __init__(
        self,
        embed_tokens,
        decoder_layers=24,
        decoder_attention_heads=16,
        max_target_positions=2048,
        embed_dim=1536,
        decoder_ffn_embed_dim=6144,
        no_scale_embedding=True,
        share_decoder_input_output_embed=True,
        decoder_learned_pos=True,
        dropout=0.1,
        attention_dropout=0.1,
        activation_fn="relu",
        add_bias_kv=False,
        add_zero_attn=False,
        disable_affine_ln=False,
        disable_bias=False,
        tensor_parallel_init_model_on_gpu=False,
        full_megatron_init=False,
        megatron_init_sigma=0.006,
        truncate_init=False,
        dtype=torch.float32,
    ):
        super().__init__()
        self.register_buffer("version", torch.Tensor([3]))
        self._future_mask = torch.empty(0)
        self.embed_tokens = embed_tokens
        self.padding_idx = embed_tokens.padding_idx
        self.embed_scale = 1.0 if no_scale_embedding else math.sqrt(embed_dim)
        self.layers = nn.ModuleList()
        self.embed_positions = None

        # Positional Embeddings
        if decoder_learned_pos:
            self.embed_positions = PositionalEmbedding(
                max_target_positions,
                embed_dim,
                self.padding_idx,
                learned=True,
                full_megatron_init=full_megatron_init,
                megatron_init_sigma=megatron_init_sigma,
                truncate_init=truncate_init,
            )

        # Layers
        for _ in range(decoder_layers):
            self.layers.append(
                TransformerDecoderLayer(
                    embed_dim,
                    embed_dim,
                    dropout=dropout,
                    decoder_attention_heads=decoder_attention_heads,
                    attention_dropout=attention_dropout,
                    decoder_ffn_embed_dim=decoder_ffn_embed_dim,
                    activation_fn=activation_fn,
                    add_bias_kv=add_bias_kv,
                    add_zero_attn=add_zero_attn,
                    disable_affine_ln=disable_affine_ln,
                    disable_bias=disable_bias,
                    tensor_parallel_init_model_on_gpu=tensor_parallel_init_model_on_gpu,
                    full_megatron_init=full_megatron_init,
                    megatron_init_sigma=megatron_init_sigma,
                    truncate_init=truncate_init,
                )
            )

        self.layer_norm = LayerNorm(
            embed_dim, elementwise_affine=not disable_affine_ln
        )

        # Output projection
        if share_decoder_input_output_embed:
            self.output_projection = nn.Linear(
                embed_tokens.weight.shape[1],
                embed_tokens.weight.shape[0],
                bias=False,
            )
            self.output_projection.weight = self.embed_tokens.weight
        else:
            self.output_projection = nn.Linear(
                embed_dim,
                2048,  # Assuming vocab size matches embed_tokens
                bias=False,
            )

    def forward(
        self,
        prev_output_tokens,
        incremental_state=None,
        features_only=False,
        self_attn_padding_mask=None,
    ):
        x, _, _ = self.forward_embedding(prev_output_tokens, incremental_state)
        self_attn_mask = self.buffered_future_mask(x, prev_output_tokens)
        for layer in self.layers:
            x = layer(
                x,
                incremental_state=incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
            )
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        if not features_only:
            x = self.output_projection(x)
        return x.transpose(0, 1).contiguous()  # B x T x C

    def forward_embedding(
        self,
        tokens,
        incremental_state=None,
    ):
        positions = self.embed_positions(tokens) if self.embed_positions else None
        if incremental_state is not None:
            tokens = tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]
        embed = self.embed_scale * self.embed_tokens(tokens)
        if positions is not None:
            embed += positions
        return embed.transpose(0, 1).contiguous(), None, None  # T x B x C

    def buffered_future_mask(self, tensor, tokens):
        bsz, seq_len = tensor.size(1), tensor.size(0)
        max_pos = seq_len
        mask = torch.triu(
            fill_with_neg_inf(torch.zeros([max_pos, max_pos], device=tensor.device)),
            1,
        )
        return mask.to(tensor)

# Helper Functions and Classes
def fill_with_neg_inf(t):
    return t.float().fill_(float("-inf")).type_as(t)

def make_positions(tensor, padding_idx):
    mask = tensor.ne(padding_idx).int()
    return (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + padding_idx

class LearnedPositionalEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, padding_idx):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.max_positions = self.num_embeddings - padding_idx - 1 if padding_idx else self.num_embeddings

def PositionalEmbedding(
    num_embeddings,
    embedding_dim,
    padding_idx,
    learned=False,
    learned_sinusoidal=False,
    full_megatron_init=False,
    pos_init_scalar=1.0,
    megatron_init_sigma=None,
    truncate_init=False,
):
    def _init_emb(tensor, sigma):
        if sigma <= 1e-8:
            return nn.init.zeros_(tensor)
        if truncate_init:
            return nn.init.trunc_normal_(tensor, 0.0, sigma, -3*sigma, 3*sigma)
        else:
            return nn.init.normal_(tensor, 0.0, sigma)

    if learned:
        num_embeddings += padding_idx + 1 if padding_idx else 0
        m = LearnedPositionalEmbedding(num_embeddings, embedding_dim, padding_idx)
        sigma = megatron_init_sigma * pos_init_scalar if full_megatron_init else embedding_dim**-0.5 * pos_init_scalar
        _init_emb(m.weight, sigma)
        if padding_idx is not None:
            nn.init.constant_(m.weight[padding_idx], 0)
    else:
        m = SinusoidalPositionalEmbedding(embedding_dim, padding_idx)
    return m

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim, padding_idx):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        weights = self.get_embedding(1024, embedding_dim, padding_idx)
        self.register_buffer('weights', weights)

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx):
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx] = 0
        return emb

    def forward(self, positions):
        max_pos = self.weights.size(0)
        if positions.max() >= max_pos or positions.min() < -max_pos:
            # Dynamically extend embeddings (placeholder logic)
            pass
        return self.weights.index_select(0, positions.view(-1)).view(*positions.size(), -1)

class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim,
        decoder_embed_dim,
        dropout=0.1,
        decoder_attention_heads=8,
        attention_dropout=0.1,
        decoder_ffn_embed_dim=2048,
        activation_fn="relu",
        add_bias_kv=False,
        add_zero_attn=False,
        disable_affine_ln=False,
        disable_bias=False,
        tensor_parallel_init_model_on_gpu=False,
        full_megatron_init=False,
        megatron_init_sigma=0.006,
        truncate_init=False,
    ):
        super().__init__()
        self.self_attn = MultiheadAttention(
            decoder_embed_dim,
            decoder_attention_heads,
            dropout=attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=True,
            bias=not disable_bias,
        )
        self.nh = decoder_attention_heads
        self.head_dim = decoder_embed_dim // self.nh
        self.self_attn_layer_norm = LayerNorm(decoder_embed_dim, elementwise_affine=not disable_affine_ln)
        self.fc1 = nn.Linear(decoder_embed_dim, decoder_ffn_embed_dim, bias=not disable_bias)
        self.activation = ActivationFn(activation_fn, decoder_embed_dim, decoder_ffn_embed_dim)
        self.fc2 = nn.Linear(decoder_ffn_embed_dim, decoder_embed_dim, bias=not disable_bias)
        self.final_layer_norm = LayerNorm(decoder_embed_dim, elementwise_affine=not disable_affine_ln)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, incremental_state, self_attn_mask, self_attn_padding_mask):
        residual = x
        x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            attn_mask=self_attn_mask,
        )
        x = self.dropout(x)
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        x = self.dropout(x)
        return x + residual

class MultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        initialize_params_on_gpu=False,
        dtype=torch.float,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.self_attention = self_attention
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, key_padding_mask, incremental_state, attn_mask):
        tgt_len, bsz, embed_dim = query.size()
        q = self.q_proj(query) * self.scaling
        k = self.k_proj(key)
        v = self.v_proj(value)
        q = q.view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        if key_padding_mask is not None:
            attn_weights = attn_weights.view(
                bsz, self.num_heads, tgt_len, -1
            ).masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, -1)
        attn_weights_float = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_weights_float.type_as(attn_weights))
        attn = torch.bmm(attn_probs, v)
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        return self.out_proj(attn), None

class ActivationFn(nn.Module):
    def __init__(self, name, embed_dim, ffn_dim):
        super().__init__()
        self.fn = self._get_activation(name)

    def forward(self, x):
        return self.fn(x)

    def _get_activation(self, name):
        if name == "relu":
            return F.relu
        elif name == "gelu":
            return F.gelu
        elif name == "relu_squared":
            return lambda x: F.relu(x) ** 2
        else:
            return lambda x: x

def LayerNorm(normalized_shape, **kwargs):
    return nn.LayerNorm(normalized_shape, **kwargs)

def my_model_function():
    embed_dim = 1536
    vocab_size = 2048
    embed_tokens = nn.Embedding(vocab_size, embed_dim, padding_idx=-1)
    return MyModel(
        embed_tokens,
        decoder_layers=24,
        decoder_attention_heads=16,
        max_target_positions=2048,
        embed_dim=embed_dim,
        decoder_ffn_embed_dim=embed_dim *4,
        no_scale_embedding=True,
        share_decoder_input_output_embed=True,
        decoder_learned_pos=True,
        dropout=0.1,
    )

def GetInput():
    return torch.randint(0, 2048, (2, 10), dtype=torch.long)

