import torch.nn as nn

import torch
import torch.utils.checkpoint
import torch._dynamo
torch._dynamo.config.verbose=True
import argparse
import torch.utils.benchmark as benchmark


class myModel(torch.nn.Module):
    def __init__(self, grad_checkpoint):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(128, 3, kernel_size=3, padding=1)
        self.checkpoint = grad_checkpoint

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        if self.checkpoint:
            x = torch.utils.checkpoint.checkpoint(self.conv2, x)
        else:
            x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = self.conv3(x)
        return x


def run_forward(model_, x):
    out = model_(x)


def run(grad_checkpoint):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = myModel(grad_checkpoint).to(device)
    x = torch.randn((2, 3, 640, 256), device=device)

    model_opt = torch.compile(model, mode="reduce-overhead")
    num_threads = torch.get_num_threads()
    t = benchmark.Timer(
        stmt='optim(x)',
        globals={'optim': model_opt, 'x': x}, # When 'optim': model then it works
        num_threads=num_threads,
        label="Average Run Duration",
    )
    print(t.timeit(100))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grad_checkpoint", action='store_true')
    args = parser.parse_args()
    run(args.grad_checkpoint)

from torch.distributed.algorithms._checkpoint import checkpoint_wrapper
...

def apply_activation_checkpointing(model: torch.nn.Module) -> torch.nn.Module:
    class_to_checkpoint = ... # class reference to the entire decoder block
    def check_fn(submodule: Any) -> bool:
            return isinstance(submodule, class_to_checkpoint)

    wrapper = functools.partial(checkpoint_wrapper.checkpoint_wrapper,
                                checkpoint_impl=checkpoint_wrapper.CheckpointImpl.NO_REENTRANT,
                                preserve_rng_state=False)
    checkpoint_wrapper.apply_activation_checkpointing(
        model,  # pylint: disable=protected-access
        checkpoint_wrapper_fn=wrapper,
        check_fn=check_fn)

    return model

import functools
from typing import Any

import torch
from absl import app, logging
from torch.distributed.algorithms._checkpoint import checkpoint_wrapper


class FlashCausalAttention(torch.nn.Module):

    def __init__(self, *, n_heads: int, ndim: int):
        super().__init__()

        self.head_dim = ndim // n_heads
        self.dims = (ndim, ndim, ndim)
        self.q_heads = self.k_heads = self.v_heads = n_heads
        self.c_attn = torch.nn.Linear(ndim, sum(self.dims))
        self.c_proj = torch.nn.Linear(ndim, ndim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.dims, dim=2)

        q = q.view(B, T, self.q_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.k_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.v_heads, self.head_dim).transpose(1, 2)

        with torch.backends.cuda.sdp_kernel(enable_flash=True,
                                            enable_math=False,
                                            enable_mem_efficient=False):
            y = torch.nn.functional.scaled_dot_product_attention(q,
                                                                 k,
                                                                 v,
                                                                 dropout_p=0.0,
                                                                 is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class SwigluMlp(torch.nn.Module):

    def __init__(self, *, ndim: int, hidden_dim: int):
        super().__init__()

        self.c_fc = torch.nn.Linear(ndim, hidden_dim)
        self.linear_v = torch.nn.Linear(ndim, hidden_dim)
        self.c_proj = torch.nn.Linear(hidden_dim, ndim)
        self.swiglu_impl = torch.nn.functional.silu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.swiglu_impl(self.c_fc(x))
        x = g * self.linear_v(x)
        return self.c_proj(x)


class DecoderBlock(torch.nn.Module):

    def __init__(self, *, n_heads: int, ndim: int):
        super().__init__()

        self.ln_1 = torch.nn.LayerNorm(ndim)
        self.attn = FlashCausalAttention(n_heads=n_heads, ndim=ndim)
        self.ln_2 = torch.nn.LayerNorm(ndim)
        self.mlp = SwigluMlp(ndim=ndim, hidden_dim=4 * ndim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x + self.attn(self.ln_1(x))
        return h + self.mlp(self.ln_2(h))


class BaseGpt(torch.nn.Module):

    def __init__(self, *, n_layers: int, n_heads: int, ndim: int, vocab_size: int):
        super().__init__()

        decoder_blocks = torch.nn.ModuleList(
            [DecoderBlock(n_heads=n_heads, ndim=ndim) for _ in range(n_layers)])
        self.transformer = torch.nn.ModuleDict({
            "wte": torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=ndim),
            "decoder_blocks": decoder_blocks,
            "ln_f": torch.nn.LayerNorm(ndim)
        })
        self.lm_head = torch.nn.Linear(ndim, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.transformer.wte(input_ids)
        for decoder_block in self.transformer.decoder_blocks:
            x = decoder_block(x)
        x = self.transformer.ln_f(x)
        return self.lm_head(x)


class TrainingGpt(torch.nn.Module):

    def __init__(self, base_model: torch.nn.Module):
        super().__init__()
        self.base_model = base_model
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = self.base_model(input_ids)
        return self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))


def apply_activation_checkpointing(model: torch.nn.Module) -> torch.nn.Module:

    def _check_fn(submodule: Any) -> bool:
        return isinstance(submodule, DecoderBlock)

    wrapper = functools.partial(
        checkpoint_wrapper.checkpoint_wrapper,
        checkpoint_impl=checkpoint_wrapper.CheckpointImpl.NO_REENTRANT,
        preserve_rng_state=False)
    checkpoint_wrapper.apply_activation_checkpointing(model,
                                                      checkpoint_wrapper_fn=wrapper,
                                                      check_fn=_check_fn)

    return model


def main(argv):
    del argv  # Unused.

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vocab_size, max_seq_length = 50304, 1024

    model = TrainingGpt(
        BaseGpt(n_layers=12, n_heads=12, ndim=768, vocab_size=vocab_size))
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters())
    logging.info("Model: {} is initialized with {:.2f} B weights.".format(
        model,
        sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9))

    logging.info("Compiling model...")
    model = torch.compile(model)
    logging.info("Applying activation checkpointing...")
    model = apply_activation_checkpointing(model)

    random_x = torch.randint(0, vocab_size, (1, max_seq_length), device=device)
    random_y = torch.randint(0, vocab_size, (1, max_seq_length), device=device)
    with torch.amp.autocast(device_type=device.type, enabled=True,
                            dtype=torch.bfloat16):
        loss = model(random_x, random_y)
    logging.info("Loss: %s", loss)

    logging.info("Running backward pass...")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    logging.info("All good.")


if __name__ == '__main__':
    app.run(main)