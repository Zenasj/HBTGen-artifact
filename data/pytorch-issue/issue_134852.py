import torch.nn as nn

import torch
import torch.nn.functional as F
from functools import lru_cache
from torch.nn.attention.flex_attention import (
    create_block_mask,
    flex_attention as original_flex_attention,  # 原始的flex_attention函数
)

torch.set_default_device("cuda")
torch.manual_seed(0)

# 编译flex_attention
flex_attention = torch.compile(original_flex_attention, dynamic=False)

@lru_cache
def create_block_mask_cached(score_mod, B, H, M, N, device="cuda"):
    block_mask = create_block_mask(score_mod, B, H, M, N, device=device,KV_BLOCK_SIZE=1024,Q_BLOCK_SIZE=1024)
    return block_mask

def prefix_lm_causal_mask(b, h, q_idx, kv_idx):
    seq_len = 93696
    block_size = seq_len // 8
    frame_size = seq_len // 2
    row_mask = q_idx % frame_size < block_size
    return row_mask

# 设置参数
B = 2
H = 24
S = 93696
D = 128

# 重置CUDA内存统计
torch.cuda.reset_peak_memory_stats()

# 记录block_mask生成前的内存状态
initial_allocated_memory = torch.cuda.memory_allocated()
initial_reserved_memory = torch.cuda.memory_reserved()

# 创建 CUDA 事件来测量时间
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

# 同步并开始计时
torch.cuda.synchronize()
start_event.record()

# 生成block_mask
block_mask = create_block_mask_cached(prefix_lm_causal_mask, 1, 1, S, S, device="cuda")

# 同步并停止计时
end_event.record()
torch.cuda.synchronize()
print(block_mask)
print(id(block_mask))

# 记录block_mask生成后的内存使用情况
after_allocated_memory = torch.cuda.memory_allocated()
after_reserved_memory = torch.cuda.memory_reserved()

# 计算block_mask的内存占用
block_mask_allocated_memory = after_allocated_memory - initial_allocated_memory
block_mask_reserved_memory = after_reserved_memory - initial_reserved_memory

print(f"BlockMask allocated memory: {block_mask_allocated_memory / (1024 ** 2):.2f} MB")
print(f"BlockMask reserved memory: {block_mask_reserved_memory / (1024 ** 2):.2f} MB")
print(f"BlockMask creation time: {start_event.elapsed_time(end_event):.4f} ms")

# 创建查询、键和值张量
query = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16, requires_grad=False)
key = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16, requires_grad=False)
value = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16, requires_grad=False)

# 重置CUDA内存统计
torch.cuda.reset_peak_memory_stats()

# 记录前向传播前的内存状态
initial_allocated_memory = torch.cuda.memory_allocated()
initial_reserved_memory = torch.cuda.memory_reserved()

# 同步并开始计时
torch.cuda.synchronize()
start_event.record()

# 运行一次前向传播
output = flex_attention(query, key, value, score_mod=None, block_mask=block_mask)

# 同步并停止计时
end_event.record()
torch.cuda.synchronize()

# 记录前向传播后的内存使用情况
peak_allocated_memory = torch.cuda.max_memory_allocated()
peak_reserved_memory = torch.cuda.max_memory_reserved()
end_allocated_memory = torch.cuda.memory_allocated()
end_reserved_memory = torch.cuda.memory_reserved()

# 打印前向传播时间和内存使用
print(f"Forward pass time: {start_event.elapsed_time(end_event):.4f} ms")
print(f"Initial allocated memory: {initial_allocated_memory / (1024 ** 2):.2f} MB")
print(f"Peak allocated memory: {peak_allocated_memory / (1024 ** 2):.2f} MB")
print(f"End allocated memory: {end_allocated_memory / (1024 ** 2):.2f} MB")
print(f"Memory allocated by operation: {(end_allocated_memory - initial_allocated_memory) / (1024 ** 2):.2f} MB")

print(f"Initial reserved memory: {initial_reserved_memory / (1024 ** 2):.2f} MB")
print(f"Peak reserved memory: {peak_reserved_memory / (1024 ** 2):.2f} MB")
print(f"End reserved memory: {end_reserved_memory / (1024 ** 2):.2f} MB")
print(f"Memory reserved by operation: {(end_reserved_memory - initial_reserved_memory) / (1024 ** 2):.2f} MB")

# 清理内存
del query, key, value, output
torch.cuda.empty_cache()

[tasklist]
### Tasks