import torch
import torch.nn as nn


class TorchCausalAttention(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.mhsa = nn.MultiheadAttention(
            embed_dim=32,
            num_heads=1,
            dropout=0.5,
            bias=True,
            batch_first=True,
            device=device,
        )

        self.register_buffer("mask", torch.empty((2, 2), device=device))
        self.mask_initialized = False
        self.mhsa.out_proj._is_residual = True  # type: ignore

    def _fill_causal_attn_mask(self):
        assert isinstance(self.mask, torch.Tensor)  # for type checking
        torch.full(size=self.mask.shape, fill_value=float("-inf"), out=self.mask)
        torch.triu(input=self.mask, diagonal=1, out=self.mask)

    def forward(self, x, key_padding_mask):
        # Two important disclaimers
        # 1. Torch uses additive attention. If your attn_mask/key_padding mask is a float tensor, it will add the floats
        #   directly to your attention matrix. If they are boolean masks, True will be converted to -inf before adding the
        #   mask to your attentions. See https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html#torch.nn.MultiheadAttention.forward
        #   Basically True/-inf indicates tokens we do not want to attend to.
        #
        # 2. This is is the exact opposite behavior of Huggingface's tokenizers, which use the convention that True denotes tokens
        #   we do want to attend to. See https://huggingface.co/docs/transformers/glossary#attention-mask
        #
        if not self.mask_initialized:
            self._fill_causal_attn_mask()
            self.mask_initialized = True

        return self.mhsa(
            x,
            x,
            x,
            attn_mask=self.mask,
            key_padding_mask=~key_padding_mask,
            need_weights=True,
        )


device = "cuda:0"
model = TorchCausalAttention(device=device)
model = torch.compile(model)
x = torch.randn(16, 2, 32).to(device=device)
key_padding_mask = torch.ones(16, 2, dtype=torch.bool).to(device=device)
out = model(x, key_padding_mask)
print(out)