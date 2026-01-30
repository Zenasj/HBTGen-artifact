import torch

def foo(p: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    p = torch.sigmoid(p)
    result = p ** gamma
    return result

script_foo = torch.jit.script(foo)
dtype = torch.half # No error if this is torch.float32

a = torch.rand((2,2), dtype=dtype, device="cuda") # Only error in gpu

print(script_foo(a)) # This call succeed
print("-----  FIRST CALL SUCCESS! ------")
print(script_foo(a)) # This call failed!
print("----- SECOND CALL FAILED! YOU WILL NOT SEE THIS! ------")