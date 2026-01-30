import torch
input_test = torch.tensor([(-6.254598811526374e+17+0j)], dtype=torch.complex128)

out_cpu = torch.acos(input_test)
print("Output from cpu:", out_cpu)

input_test_gpu = input_test.cuda()
out_gpu = torch.acos(input_test_gpu)
print("Output from gpu:", out_gpu)