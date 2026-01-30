import torch
import torchdynamo

toy_sum = lambda tlist: sum(tlist)

def toy_example(tlist):
    ret_val = 0
    for _ in range(5):
        ret_val += toy_sum(tlist)
    return ret_val


tlist = [torch.ones(10), torch.ones(10), torch.ones(10)]

print("with dynamo")
with torchdynamo.optimize("eager"):
    for i in range(5):
        if i == 2:  # alter `toy_sum` after the third iteration
            toy_sum = lambda tlist: tlist[0]
        print(toy_example(tlist))


print("without dynamo")
toy_sum = lambda tlist: sum(tlist)
for i in range(5):
    if i == 2:  # alter `toy_sum` after the third iteration
        toy_sum = lambda tlist: tlist[0]
    print(toy_example(tlist))

import torch


tlist = [torch.ones(10), torch.ones(10), torch.ones(10)]

def test():
    toy_sum = lambda tlist: sum(tlist)
    def toy_example(tlist):
        ret_val = 0
        for _ in range(5):
            ret_val += toy_sum(tlist)
        return ret_val
    for i in range(5):
        if i == 2:  # alter `toy_sum` after the third iteration
            toy_sum = lambda tlist: tlist[0]
        print(toy_example(tlist))
    return toy_sum


print("compiled run")
out = torch.compile(test, backend="eager", fullgraph=False)()


print("eager run")
test()