import torch

def main():
    def fun(x, y):
        return torch.cat([x, y], dim=0)

    fun = torch.jit.trace(fun, (torch.ones([4, 4], requires_grad=True),
                                torch.zeros([4, 4], requires_grad=True)))
    out = torch._C._jit_differentiate(fun.graph)
    return out


if __name__ == '__main__':
    main()