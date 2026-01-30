import contextlib
import torch
import torch.nn.functional as F
import torch.nn as nn

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

contexts = {
    "no context": contextlib.nullcontext(),
}
if device.type == "cpu":
    contexts["SDPBackend.FLASH_ATTENTION"] = nn.attention.sdpa_kernel(nn.attention.SDPBackend.FLASH_ATTENTION)
    contexts["SDPBackend.MATH"] = nn.attention.sdpa_kernel(nn.attention.SDPBackend.MATH)

with torch.device(device):
    for name, ctx in contexts.items():
        with ctx:
            query = torch.randn(2, 4, 16, 8)
            key = torch.randn(2, 4, 16, 8)
            value = torch.randn(2, 4, 16, 8)

            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            assert torch.isfinite(hidden_states).all()

            query += torch.nan
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            try:
                assert torch.isnan(hidden_states).all()
                print(f"Succeeded with device: {device}, torch version: {torch.__version__}, context: {name}")
            except AssertionError:
                print(f"Failed    with device: {device}, torch version: {torch.__version__}, context: {name}")