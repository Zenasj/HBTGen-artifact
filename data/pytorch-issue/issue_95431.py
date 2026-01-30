import torch


def main():
    x, y = [[torch.randn(1, requires_grad=True) for _ in range(2)] for _ in range(2)]
    outputs = torch._foreach_pow(x, y)
    print(f"{len(outputs[0].grad_fn._saved_self) = }")


if __name__ == "__main__":
    main()