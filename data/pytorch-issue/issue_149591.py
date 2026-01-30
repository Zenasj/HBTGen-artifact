import torch

def reproduce_bug():
    m, n, k = 50000, 8193, 10
    H_aug = torch.randn(n, m, dtype=torch.float32)  #Change to torch.float64 here
    T = torch.randn(n, k, dtype=torch.float32)  #Change to torch.float64 here
    W_aug = torch.pinverse(H_aug) @ T
    print("W_aug shape:", W_aug.shape)

if __name__ == "__main__":
    reproduce_bug()