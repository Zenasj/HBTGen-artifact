import torch

def repro():
    def fn():
        d = dict({"a": 1, "b": "2", "c": torch.tensor(3)})
        return d.items()

    opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
    ref = fn()
    res = opt_fn()

    print(f"Eager: {ref}")
    print(f"Dynamo: {res}")