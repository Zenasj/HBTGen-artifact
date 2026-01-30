import torch

def dequant_int4(data, scale, shape):
    data = torch.stack([(data >> 4) & 0b1111, data & 0b1111], dim=-1)  # unpacking

    data = data.float() - 8  # shift [0, 15] to [-8, 7]
    data = data.view(scale.shape[0], -1) * scale.view(-1, 1)
    return data.view(shape)

def quantize_int4(data, block_size=128):
    scale = 7.5 / data.view(-1, block_size).abs().amax(-1)
    data = (data.view(-1, block_size) * scale.view(-1, 1)).clip(-8, 7)
    data = (data + 8).to(torch.uint8).view(-1)  # shift [-8, 7] to [0, 15]

    data = (data[::2] << 4) | data[1::2]  # packing
    return data, scale

def single_sgd(p, grad, step, exp_avg_int4, exp_avg_scale, lr, beta1, beta2, eps):
    exp_avg_fp32 = dequant_int4(exp_avg_int4, exp_avg_scale, p.shape)

    exp_avg_fp32 = exp_avg_fp32.lerp(grad, 1 - beta1)
    bias_correction1 = 1 - beta1 ** step
    p.add_(-lr * exp_avg_fp32 / bias_correction1)

    new_int4, new_scale = quantize_int4(exp_avg_fp32)
    exp_avg_int4.copy_(new_int4)
    exp_avg_scale.copy_(new_scale)

def multi_sgd(params, lr, beta1, beta2, eps):
    for p, grad, step, exp_avg_int4, exp_avg_scale in params:
        single_sgd(p, grad, step, exp_avg_int4, exp_avg_scale, lr, beta1, beta2, eps)


lr = 1e-4
beta1 = 0.9
beta2 = 0.999
eps = 1e-8
step = torch.tensor(100.0, device="cuda")

shapes = [(4096,), (4096, 4096), (4096, 16384)] * 10

params = []
for shape in shapes:
    p = torch.randn(shape, device="cuda")
    grad = torch.randn(shape, device="cuda")
    exp_avg = torch.randn(shape, device="cuda")
    exp_avg_int4, exp_avg_scale = quantize_int4(exp_avg)
    params.append((p, grad, step, exp_avg_int4, exp_avg_scale))


print(torch.__version__)

torch.cuda.reset_peak_memory_stats()
for p, grad, step, exp_avg_int4, exp_avg_scale in params:
    torch.compile(single_sgd, fullgraph=True)(p, grad, step, exp_avg_int4, exp_avg_scale, lr, beta1, beta2, eps)
print(f"Compile inner function: {torch.cuda.max_memory_allocated() / 1e9} GB")

torch.cuda.reset_peak_memory_stats()
torch.compile(multi_sgd, fullgraph=True)(params, lr, beta1, beta2, eps)
print(f"Compile outer function: {torch.cuda.max_memory_allocated() / 1e9} GB")