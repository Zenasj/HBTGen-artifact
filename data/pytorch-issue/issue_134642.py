import torch.nn as nn

import torch
from tqdm import tqdm
from transformers import LlamaConfig, LlamaForCausalLM


def get_loss(model: LlamaForCausalLM, batch: torch.Tensor):
    logits = model(batch).logits[:, :-1].flatten(0, 1)
    labels = batch[:, 1:].flatten()
    return torch.nn.functional.cross_entropy(logits, labels)


if __name__ == "__main__":
    seq_len = 2048
    config = LlamaConfig(
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=12,
        num_attention_heads=8,
        max_position_embeddings=seq_len,
        use_cache=False,
    )
    model = LlamaForCausalLM(config).bfloat16().cuda()
    model.gradient_checkpointing_enable()

    optim = torch.optim.AdamW(model.parameters())

    n_steps = 10_000
    step = 0
    log_interval = 50
    bsize = 4
    pbar = tqdm(total=n_steps, dynamic_ncols=True)
    model.train()

    while step < n_steps:
        batch = torch.randint(0, config.vocab_size, (bsize, seq_len), device="cuda")
        loss = torch.compile(get_loss)(model, batch)
        loss.backward()
        optim.step()
        optim.zero_grad()

        step += 1
        pbar.update()