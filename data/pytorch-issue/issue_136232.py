import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from modules import RoPE

@lru_cache
def create_block_mask_cached(mask_mod, B, H, Q_LEN, KV_LEN, device, BLOCK_SIZE=64):
    print("[create_block_mask_cached] expensive call")
    return create_block_mask(mask_mod, B=B, H=H, Q_LEN=Q_LEN, KV_LEN=KV_LEN, 
                             device=device, BLOCK_SIZE=BLOCK_SIZE)

@lru_cache
def create_sliding_window_mask_cached(sliding_window_size, prefix_size=0):
    def sliding_window_mask_mod(b_idx, h_idx, q_idx, kv_idx):
        # queries attend to prefix and neighbouring keys
        return (kv_idx < prefix_size) | ((q_idx - kv_idx).abs() <= sliding_window_size)
    return sliding_window_mask_mod


flex_attention = torch.compile(flex_attention, dynamic=False)

class Attn(nn.Module):
    def __init__(self, name, d_model, H, d_head=64, num_sinks=16, sliding_window_size=64, **kwargs):
        super().__init__()
        assert H % d_head == 0
        self.d_model, self.d_head, self.name = d_model, d_head, name
        self.QKV = nn.Linear(d_model, 3*H)
        self.O = nn.Linear(H, d_model)
        self.rope = RoPE(self.d_head)
        self.num_sinks = num_sinks = max(num_sinks, 0)
        if num_sinks > 0:
            self.sinks = nn.Parameter(torch.randn(H // d_head, num_sinks, d_head) / d_head**.5)
        self.sliding_window_size = sliding_window_size
        
    def forward(self, u, use_flex=True):
        """[... L d_model]"""
        shape, u = u.shape, u.flatten(0,-3)   # [B L d_model]
        
        B, L, S = *u.shape[:2], self.num_sinks
        
        form_heads = lambda x: x.unflatten(-1, (-1, self.d_head)).transpose(-3,-2)  # [B L n*dh] -> [B n L dh]
        Q, K, V = self.QKV(u).chunk(3,-1)                                 # [B L H]
        Q, K, V = map(form_heads, (Q, K, V))                              # [B n L dh]
        Q, K = map(self.rope, (Q, K))                                     # [B n L dh]
        
        if S:
            sinks = self.sinks.unsqueeze(0).expand(B,-1,-1,-1)
            Q, K, V = (torch.cat((sinks, x), dim=-2) for x in (Q, K, V))  # [B n S+L dh]
        
        if self.sliding_window_size > 0 and self.sliding_window_size < L:
            if use_flex:
                mask_mod = create_sliding_window_mask_cached(self.sliding_window_size, prefix_size=S)
                block_mask = create_block_mask_cached(mask_mod, B=None, H=None, Q_LEN=L+S, KV_LEN=L+S, device=Q.device)
                O = flex_attention(Q, K, V, block_mask=block_mask)
            else:
                q_idx = torch.arange(Q.shape[-2], device=Q.device).view(-1,1)
                kv_idx = torch.arange(K.shape[-2], device=Q.device).view(1,-1)
                attn_mask = (kv_idx < S) | ((q_idx - kv_idx).abs() <= self.sliding_window_size)
                O = F.scaled_dot_product_attention(Q, K, V, attn_mask)    
        else:
            O = F.scaled_dot_product_attention(Q, K, V)                   # [B n S+L dh]        
        
        y = O[:,:,-L:].transpose(-3,-2).flatten(-2,-1)                    # [B L H]
        y = self.O(y)                                                     # [B L d_model]
        
        return y.view(shape)