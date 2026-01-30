Python
import torch

def main():
    a = torch.rand(16, 16, device="cuda", dtype=torch.bfloat16)
    b = torch.rand(16, device="cuda", dtype=torch.float32)

    @torch.compile(fullgraph = True)
    def func(a, b):
        abs_max = torch.abs(a).max()
        b[0] = abs_max.to(a.dtype)

    func(a, b)

if __name__ == '__main__':
    main()