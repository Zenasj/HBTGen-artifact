import random

import logging
import math

import numpy as np
import torch
import torch._dynamo
import torch._dynamo.config
import torch._inductor
import torch._inductor.config
import torch.nn as nn
from numpy import sign
from transformers import T5Tokenizer


def pad_mask(inp: torch.Tensor, pad: int):
    padmask = inp == pad
    mask: torch.BoolTensor = padmask.unsqueeze(-1) == padmask.unsqueeze(-2)
    return mask


class Alibi(nn.Module):
    """
    Attention Linear Bias layer for sequence models, as in https://arxiv.org/pdf/2108.12409.pdf.
    """

    def __init__(self, nheads, max_scale=0.5, min_scale=1 / (2**8)):
        super(Alibi, self).__init__()
        self.nheads = nheads
        start = math.log2(max_scale)
        end = math.log2(min_scale)
        self.register_buffer(
            "scales",
            2
            ** torch.arange(start, end + 1e-6 * sign(end - start), (end - start) / (nheads - 1)).view(1, nheads, 1, 1),
        )

    def forward(self, qlen, klen):
        # Automatically allocates on chosen cuda
        device = self.scales.device
        q_pos = torch.arange(qlen, dtype=torch.long).to(device)
        k_pos = torch.arange(klen, dtype=torch.long).to(device)

        # rel_pos: qlen x klen
        rel_pos = k_pos[None, :] - q_pos[:, None]
        values = rel_pos.abs().neg().unsqueeze(0).unsqueeze(0)

        return values * self.scales


class ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.emb = nn.Embedding(32100, 1024)
        self.head = nn.Linear(1024, 32100)

        self.relpos_bias = Alibi(16, max_scale=2**-1, min_scale=2**-8)

        self.dec_process = nn.ModuleList(
            [nn.TransformerDecoderLayer(1024, 16, 4096, batch_first=True, norm_first=True) for _ in range(24)]
        )

    def forward(self, enc_out, dec_in):
        dec_mask = pad_mask(dec_in, 0).to(dtype=torch.float32)
        dec_mask = dec_mask.tril(diagonal=0)
        dec_bias = self.relpos_bias(dec_in.size(1), dec_in.size(1)).tril(diagonal=0)
        dec_mask = dec_mask + dec_bias[0]
        emb = self.emb(dec_in)

        for layer in self.dec_process:
            emb = layer(emb, enc_out, tgt_mask=dec_mask)

        return self.head(emb)


# Set random seeds and deterministic pytorch for reproducibility
torch.manual_seed(42)  # pytorch random seed
np.random.seed(42)  # numpy random seed
torch.backends.cudnn.deterministic = True
# torch.set_float32_matmul_precision("high")
torch._dynamo.config.log_level = logging.INFO
torch._dynamo.config.verbose = False
torch._dynamo.config.cache_size_limit = 2048

pad_token = 0
model = ToyModel().cuda().eval()
model_opt = torch.compile(model, dynamic=True)
tokenizer = T5Tokenizer.from_pretrained("t5-large", model_max_length=1138)

source = tokenizer(
    ["Summarize: this is a test"],
    max_length=512,
    padding="max_length",
    return_tensors="pt",
)

torch.set_printoptions(threshold=10000)

input_test = torch.ones((1, 1), dtype=torch.long).cuda() * pad_token
opt_input_test = input_test.detach().clone()
# torch._dynamo.mark_dynamic(opt_input_test, 1)
enc_inputs = source["input_ids"].cuda()
memory = torch.rand(1, 512, 1024).cuda()
print(model, memory, input_test, opt_input_test)
prof_step = 0
for i in range(8):
    output_test = model(
        memory,
        input_test,
    )
    print("Eager output size:", output_test.size())
    opt_output_test = model_opt(
        memory,
        input_test,
    )
    print("Compile output size:", opt_output_test.size())

    next_token_logits = output_test[:, -1, :]
    opt_next_token_logits = opt_output_test[:, -1, :]

    # argmax
    next_tokens = torch.argmax(next_token_logits, dim=-1)
    opt_next_tokens = torch.argmax(opt_next_token_logits, dim=-1)

    input_test = torch.cat([input_test, next_tokens[:, None]], dim=-1)
    opt_input_test = torch.cat([opt_input_test, opt_next_tokens[:, None]], dim=-1)
    preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in input_test]
    opt_preds = [
        tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in opt_input_test
    ]
    print("Current sentence: ", preds, opt_preds)
    print(
        "Output logits (eager/compile)",
        output_test,
        opt_output_test,
        output_test - opt_output_test,
        torch.sqrt(torch.sum((output_test - opt_output_test).pow(2.0))),
    )
    print(
        "Next token logits (eager/compile)",
        next_token_logits,
        opt_next_token_logits,
        next_token_logits - opt_next_token_logits,
        torch.sqrt(torch.sum((next_token_logits - opt_next_token_logits).pow(2.0))),
    )
    print("Next token (eager/compile)", next_tokens, opt_next_tokens)
    print("New inputs", input_test, opt_input_test)