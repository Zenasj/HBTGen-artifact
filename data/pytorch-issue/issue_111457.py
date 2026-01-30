import unittest
import torch
class Test(unittest.TestCase):
    def test(self):
        print(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        _dtype = torch.bfloat16 
        torch.manual_seed(0)
        # python -c "import torch; print(torch.__version__)"  : 2.0.1+cu117
        in_feature = 2048 
        output_features = [512, 2048, 4096]
        for output_feature in output_features:
            # generate data
            a = torch.randn((2048, in_feature), dtype=_dtype)
            b = torch.randn((in_feature, output_feature), dtype=_dtype)
            d = torch.randn((output_feature, in_feature), dtype=_dtype)
            res_cpu = a @ b
            res_gpu = a.to('cuda') @ b.to('cuda') 
            torch.testing.assert_close(res_cpu, res_gpu.to('cpu'), rtol=1e-3, atol=1e-1)
            res_cpu = torch.matmul(res_cpu,  d)
            res_gpu = torch.matmul(res_gpu, d.to("cuda"))
            torch.testing.assert_close(res_cpu, res_gpu.to('cpu'), rtol=1e-2, atol=1e-1)
            print("in_feature : {in_feature}, output_features:{output_features} passed")
if __name__ == '__main__':
    unittest.main()

import torch
print(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
_dtype = torch.bfloat16 
torch.manual_seed(0)
# python -c "import torch; print(torch.__version__)"  : 2.0.1+cu117
in_feature = 2048
output_features = [512, 2048, 4096]
for output_feature in output_features:
    # generate data
    a = torch.randn((512, in_feature), dtype=_dtype) / in_feature
    b = torch.randn((in_feature, output_feature), dtype=_dtype) / in_feature
    res_cpu = a @ b
    res_gpu = a.to('cuda') @ b.to('cuda') 
    diff = torch.abs(res_cpu - res_gpu.cpu()) >= 0.1
    res_double = a.double() @ b.double()
    res_double_gpu = a.to('cuda').double() @ b.to('cuda').double()

    print(res_cpu[diff])
    print(res_gpu[diff])
    print(res_double[diff])
    print(res_double_gpu[diff])

    torch.testing.assert_close(res_cpu, res_gpu.to('cpu'), rtol=1e-3, atol=1e-1)
    print("in_feature : {in_feature}, output_features:{output_features} passed")

import torch
print(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
_dtype = torch.bfloat16 
torch.manual_seed(0)
# python -c "import torch; print(torch.__version__)"  : 2.0.1+cu117
in_feature = 2048
output_features = [512, 2048, 4096]
for output_feature in output_features:
    # generate data
    a = torch.randn((512, in_feature), dtype=_dtype) / float(in_feature)
    b = torch.randn((in_feature, output_feature), dtype=_dtype) / float(in_feature)
    res_cpu = (a @ b) * (in_feature ** 2)
    res_gpu = (a.to('cuda') @ b.to('cuda')) * (float(in_feature) ** 2)
    diff = torch.abs(res_cpu - res_gpu.cpu()) >= 0.1
    res_double = (a.double() @ b.double()) * (float(in_feature) ** 2)
    res_double_gpu = (a.to('cuda').double() @ b.to('cuda').double()) * (in_feature ** 2)

    print(res_cpu[diff])
    print(res_gpu[diff])
    print(res_double[diff])
    print(res_double_gpu[diff])

    torch.testing.assert_close(res_cpu, res_gpu.to('cpu'), rtol=1e-3, atol=1e-1)
    print("in_feature : {in_feature}, output_features:{output_features} passed")