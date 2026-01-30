import torch

def demo():
    # Input tensors that are generated randomly
    torch.manual_seed(777)
    in_self_ln39273 = (
        torch.randn(size=[128, 2501], device="cpu").ge(0)
    )

    def fwd_subgraph():
        # pytorch op calls encoded from aten functions
        res_ln39273_0, res_ln39273_1 = torch.max(
            in_self_ln39273, 1, False
        )  # traced line: 39273
        return res_ln39273_0, res_ln39273_1

    sg_callable = torch.compile(fwd_subgraph)
    with torch.amp.autocast('cpu', enabled=True, dtype=torch.bfloat16):
        res_ln39273_0, res_ln39273_1 = sg_callable()
    return {"res_ln39273_0": res_ln39273_0, "res_ln39273_1": res_ln39273_1}

print(demo())