import torch

def main():
    torch.manual_seed(42)
    A = torch.randn(1, 512, 2048).cuda()
    B = torch.randn(1, 2048, 512).cuda()
    C = A @ B

if __name__ == "__main__":
    main()