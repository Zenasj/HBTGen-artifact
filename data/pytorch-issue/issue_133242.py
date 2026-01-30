import torch

def test() -> None:
    ref_dtype = torch.bfloat16
    M, K, N = 4096, 4096, 3072
    
    input_tensor = torch.randn(M, K, device="cuda", dtype=ref_dtype, requires_grad=False)
    scale = torch.Tensor([10.0]).to("cuda")

    E4M3_MAX_POS = torch.finfo(torch.float8_e4m3fn).max
    E5M2_MAX_POS = torch.finfo(torch.float8_e5m2).max

    def test_pattern2(tensor_x_inp, scale_x):
        tensor_x = tensor_x_inp * scale_x
        tensor_x = tensor_x.clamp(min=-1 * E4M3_MAX_POS, max=E4M3_MAX_POS)
        tensor_fp8 = tensor_x.to(torch.float8_e4m3fn)

        tensor_x_t = (tensor_x_inp * scale_x).t()
        tensor_x_t = tensor_x_t.clamp(min=-1 * E4M3_MAX_POS, max=E4M3_MAX_POS)
        tensor_fp8_t = tensor_x_t.to(torch.float8_e4m3fn)
        
        tensor_fp8_t = tensor_fp8_t.contiguous().t()

        return (tensor_fp8, tensor_fp8_t)

    test_pattern = torch.compile(test_pattern2)
    tensor_fp8, tensor_fp8_t = test_pattern(input_tensor, scale)
    print(tensor_fp8.stride(), tensor_fp8_t.stride())


# TORCHINDUCTOR_PROFILE=1 TORCHINDUCTOR_PROFILE_OUTPUT=/tmp/profile.txt  TORCH_LOGS="fusion, +inductor,+schedule,output_code" TORCH_COMPILE_DEBUG=1 python test.py

def test_pattern2(tensor_x_inp, scale_x):
        tensor_x = tensor_x_inp * scale_x
        tensor_x = tensor_x.clamp(min=-1 * E4M3_MAX_POS, max=E4M3_MAX_POS)
        tensor_fp8 = tensor_x.to(torch.float8_e4m3fn)
        
        tensor_fp8_t = tensor_fp8_t.contiguous().t()

        return (tensor_fp8, tensor_fp8_t)