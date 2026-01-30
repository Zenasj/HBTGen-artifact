import torch

python
res_q = torch.quantize_per_tensor(torch.Tensor(res), outp_scale, outp_zero_point, torch.quint8)
res_q_dq = res_q.dequantize().numpy()

print(f_outp.detach().numpy()[0] - res_q_dq)