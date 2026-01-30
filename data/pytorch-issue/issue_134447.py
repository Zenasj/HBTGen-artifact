import torch

def f(a, b):
    return a + b

def main():
    x = torch.tensor([1.0])
    y = f(x, x)

if __name__ == "__main__":
    main()