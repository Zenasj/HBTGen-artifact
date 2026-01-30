import torch.nn as nn

import torch

class MyModel_2(torch.nn.Module):

    def __init__(self, max_len=8192, dim=1024):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=max_len, embedding_dim=dim)
        # Think these as a fixed-size static KV cache
        self.cached_keys = torch.nn.Parameter(torch.ones(size=(1, max_len, dim), dtype=torch.float32))
        self.cached_values = torch.nn.Parameter(torch.ones(size=(1, max_len, dim), dtype=torch.float32))

        self.max_len = max_len

    def forward(self, input_ids, attn_mask):

        q_len = input_ids.size()[1]
        # Create a mask with the target length being `self.max_len`
        _mask = torch.zeros(size=(1, q_len, self.max_len), dtype=torch.int32, device=input_ids.device).to(torch.bool)
        # Update `_mask` with the argument `attn_mask`
        _mask[:, :, :attn_mask.size()[2]] = attn_mask

        hidden = self.embedding(input_ids)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query=hidden,
            key=self.cached_keys,
            value=self.cached_values,
            attn_mask=_mask,
        )
        return attn_output

import datetime
import multiprocessing


def run(model_type, max_len, n_iter=4, warmup_run=False, log_steps=64, detailed=False):

    import torch
    if model_type == "MyModel_1":
        from model import MyModel_1 as MyModel
    elif model_type == "MyModel_2":
        from model import MyModel_2 as MyModel

    device = "cuda"
    model = MyModel(max_len=max_len, dim=16).to(device)
    model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)

    with torch.no_grad():
        for iter_idx in range(n_iter):

            torch.cuda.empty_cache()

            print(f"max_len: {max_len}") if not warmup_run else 0
            print(f"iter_idx: {iter_idx}") if not warmup_run else 0

            steps = range(max_len)
            if warmup_run:
                steps = [0, 1, max_len-2, max_len - 1]

            torch.cuda.synchronize()
            s = datetime.datetime.now()

            for idx in steps:

                if model_type == "MyModel_1":
                    input_ids = torch.arange(idx+1, dtype=torch.int32, device=device).unsqueeze(0)
                elif model_type == "MyModel_2":
                    if idx == 0:
                        input_ids = torch.arange(3, dtype=torch.int32, device=device).unsqueeze(0)
                    else:
                        input_ids = torch.tensor([idx], dtype=torch.int32, device=device).unsqueeze(0)

                q_len = input_ids.size()[1]
                attn_mask = torch.ones(size=(q_len, idx + 1), dtype=torch.int32, device=device).unsqueeze(0).to(torch.bool)

                if idx == 0:
                    torch.cuda.empty_cache()
                    memory = torch.cuda.mem_get_info()
                    used_mem_start = (memory[1] - memory[0]) / 1024 / 1024
                    print(f"\nUsed GPU memory: {used_mem_start} MB.") if not warmup_run else 0

                    m_start = torch.cuda.max_memory_allocated(device=device)

                if not detailed:
                    _ = model(input_ids, attn_mask)
                else:
                    m1 = torch.cuda.max_memory_allocated(device=device)
                    _ = model(input_ids, attn_mask)
                    memory = torch.cuda.mem_get_info()
                    used_mem = (memory[1] - memory[0]) / 1024 / 1024
                    m2 = torch.cuda.max_memory_allocated(device=device)
                    diff_mem = max(m2 - m1, 0) / 1024 / 1024

                    if detailed and not warmup_run and iter_idx in [0, 1]:
                        if idx in [0, 1, max_len-2, max_len-1] or (idx + 1) % log_steps == 0:
                            if idx == 0:
                                print("")
                            print(f"step: {str(idx + 1).zfill(4)} | `max_memory_allocated` increased (per step): {'%.4f' % round(diff_mem, 4)} MB | Used GPU increased (since this iter.): {'%.3f' % round(used_mem - used_mem_start, 3)} MB")

            torch.cuda.synchronize()
            e = datetime.datetime.now()
            m_end = torch.cuda.max_memory_allocated(device=device)
            used_mem_end = (memory[1] - memory[0]) / 1024 / 1024
            diff_mem = max(m_end - m_start, 0) / 1024 / 1024

            print(f"\nUsed GPU memory: {used_mem_end} MB.") if not warmup_run else 0

            print(f"\ntiming: {(e-s).total_seconds()}") if not warmup_run else 0
            print(f"max_memory_allocated increased: {diff_mem} MB.") if not warmup_run else 0
            print(f"Used GPU memory increased: {used_mem_end - used_mem_start} MB.") if not warmup_run else 0
            print("-" * 60) if not warmup_run and iter_idx < n_iter - 1 else 0


model_type = "MyModel_2"
log_steps = 256
detailed = True
for max_len in [2048]:
    for i in range(2):

        warmup_run = not i
        n_iter = 2 if warmup_run else 4
        print("=" * 80) if not warmup_run else 0

        p = multiprocessing.Process(target=run, args=(model_type, max_len, n_iter, warmup_run, log_steps, detailed))
        p.start()
        p.join()