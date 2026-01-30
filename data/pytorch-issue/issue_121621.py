import torch


def random_op(tensor, params):
    res = tensor.random_(**params)
    return res


if __name__ == "__main__":
    random_op = torch.compile(random_op)
    params = {"from": -10, "to": 10}
    tensor = torch.randn([2, 3])
    res = random_op(tensor, params)
    print(res)