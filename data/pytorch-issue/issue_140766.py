import torch.nn as nn

import sympy
from torch._inductor.dependencies import MemoryDep


if __name__ == "__main__":
    
    c0 = sympy.Symbol("c0")
    c1 = sympy.Symbol("c1")
    c2 = sympy.Symbol("c2")
    
    # force the new symbols to be allocated in a different memory location
    new_c0 = sympy.Symbol.__xnew__(sympy.Symbol, "c0")
    new_c1 = sympy.Symbol.__xnew__(sympy.Symbol, "c1")
    new_c2 = sympy.Symbol.__xnew__(sympy.Symbol, "c2")
    print(f"{id(c0)=}, {id(c1)=}, {id(c2)=}")
    print(f"{id(new_c0)=}, {id(new_c1)=}, {id(new_c2)=}")
    print(f"{(c0 == new_c0)=}, {(c1 == new_c1)=}, {(c2 == new_c2)=}")  # True
    print(f"{(c0 is new_c0)=}, {(c1 is new_c1)=}, {(c2 is new_c2)=}")  # False
    
    # canonical memory layout
    s0, s1, s2 = 2, 3, 4
    index = c0*s1*s2 + c1*s2 + c2
    test_dep_1 = MemoryDep("test_dep_1", index, (c0, c1, c2), (s0, s1, s2))
    test_dep_2 = MemoryDep("test_dep_2", index, (new_c0, new_c1, new_c2), (s0, s1, s2))
    print(test_dep_1)
    print(test_dep_2)
    print(f"{test_dep_1.stride1_for_last_dim()=}")  # True
    print(f"{test_dep_2.stride1_for_last_dim()=}")  # True
    
    # transposed memory layout
    s0, s1, s2 = 2, 4, 3
    index = c0*s1*s2 + c1 + c2*s1
    test_dep_1 = MemoryDep("test_dep_1", index, (c0, c1, c2), (s0, s1, s2))
    test_dep_2 = MemoryDep("test_dep_2", index, (new_c0, new_c1, new_c2), (s0, s1, s2))
    print(test_dep_1)
    print(test_dep_2)
    print(f"{test_dep_1.stride1_for_last_dim()=}")  # False
    print(f"{test_dep_2.stride1_for_last_dim()=}")  # Should be False, but it is True

import torch


class ReproModel(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(128, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 128),
        )
        self.norm = torch.nn.LayerNorm(128)
        self.attn = torch.nn.functional.scaled_dot_product_attention
    
    def forward(self, x):
        # [2, 128, 4096]
        x = x.transpose(1, 2)
        # [2, 4096, 128]
        for _ in range(2):
            x = self.forward_block(x)
        return x
    
    def forward_block(self, x):
        # x: B, H*W, C
        B = x.shape[0]
        H, W, C = 64, 64, 128
        shortcut = x
        x = self.norm(x)
        x = x.reshape(B, H, W, C)
        # B, H, W, C
        x = self.attn(x, x, x)
        x = x.reshape(
            B, H // 8, W // 8, 8, 8, -1
        )
        x = x.transpose(2, 3).reshape(B, H*W, -1)

        x = shortcut + x
        x = x + self.mlp(self.norm(x))

        return x


# ReproModel

device = "cuda"
dtype = torch.bfloat16
args = torch.randn(2, 128, 4096).to(device).to(dtype)
model = ReproModel().to(device).to(dtype)

model(args)

bs = torch.export.Dim("bs", max=12)

ep = torch.export.export(
    model,
    (args,),
    dynamic_shapes={
        "x": {0: bs},
    },
)
print(ep)
so_path = torch._inductor.aot_compile(
    ep.module(),
    args=(args,),
    options={
        "aot_inductor.output_path": "repro.so",
    },
)