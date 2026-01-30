import torch


def complex_matrix(real, imag):
    return torch.stack((
            torch.stack((real, -imag), dim=-1),
            torch.stack((imag, real), dim=-1)
        ), dim=-2)


to_complex = lambda x: torch.complex(x[..., 0, 0], x[..., 1, 0])


def block_dft(input):    
    input_block = input.view(16, 16, -1)
    last_dim_size = input_block.shape[-1]
    
    tau = 2 * torch.pi
    range_16 = torch.arange(16, device=input.device)
    range_large = torch.arange(16 * last_dim_size, device=input.device)
    range_small = torch.arange(last_dim_size, device=input.device)
    
    dft_real = torch.cos(-(range_16.unsqueeze(-1) * range_16) / 16 * tau)
    dft_imag = torch.sin(-(range_16.unsqueeze(-1) * range_16) / 16 * tau)
    dft = complex_matrix(dft_real, dft_imag)
    
    dft_small_real = torch.cos(-(range_small.unsqueeze(-1) * range_small) / last_dim_size * tau)
    dft_small_imag = torch.sin(-(range_small.unsqueeze(-1) * range_small) / last_dim_size * tau)
    dft_small = complex_matrix(dft_small_real, dft_small_imag)
    
    twid_real = torch.cos(-(range_16.unsqueeze(-1) * range_large) / (16 * 16 * last_dim_size) * tau).reshape(16, 16, -1)
    twid_imag = torch.sin(-(range_16.unsqueeze(-1) * range_large) / (16 * 16 * last_dim_size) * tau).reshape(16, 16, -1)
    twid = complex_matrix(twid_real, twid_imag)
    
    twid_small_real = torch.cos(-(range_16.unsqueeze(-1) * range_small) / (16 * last_dim_size) * tau)
    twid_small_imag = torch.sin(-(range_16.unsqueeze(-1) * range_small) / (16 * last_dim_size) * tau)
    twid_small = complex_matrix(twid_small_real, twid_small_imag)
    
    input_block = complex_matrix(input_block, torch.zeros_like(input_block))
    
    return torch.einsum('xyzAB,xfBC,ygDE,zhFG,fyzCD,gzEF->hgfAG', 
                        input_block, 
                        dft, dft, dft_small, 
                        twid, twid_small).flatten(0, 2)


input = torch.rand(1024)
input_f = torch.fft.fft(input)

assert torch.allclose(input_f, to_complex(block_dft(input)), rtol=1e-3, atol=1e-3)  # this will pass

block_dft_compiled = torch.compile(block_dft)
assert torch.allclose(input_f, to_complex(block_dft_compiled(input)), rtol=1e-3, atol=1e-3)  # this will fail