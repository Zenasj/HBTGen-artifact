import torch
lib=torch.mps.compile_shader("""
kernel void do_sum(device int* out, constant int* in, uint idx [[thread_position_in_grid]]) {
  out[idx] = metal::simd_shuffle_down(in[idx], 8);
}
""")
x=torch.arange(22, device='mps', dtype=torch.int32)
y=torch.empty_like(x)
lib.do_sum(y, x)
print(y)