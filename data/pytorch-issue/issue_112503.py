import torch

def fn(input1, input2):
    res = input1.bitwise_and_(input2)
    return res

if __name__ == "__main__":
    input1 = torch.randint(1, 10, (2, 3))
    input2 = True
    f = torch.compile(fn)
    f(input1, input2)