import pyarrow as pa
import torch


def test():
    n = 100_000
    my_arrow = pa.Table.from_pydict(
        {f"c{i}": [float(x) for x in range(n)] for i in range(10)})
    torch_tensors = [torch.tensor(c.to_numpy()) for c in my_arrow.columns]

    def test_torch(tensors):
        t0 = tensors[0]
        r = t0
        for idx, t in enumerate(tensors):
            r += (t * t + idx) / 2
        return r

    test_torch_compiled = torch.compile(test_torch)
    result = test_torch_compiled(torch_tensors)
    print(result)


if __name__ == '__main__':
    test()