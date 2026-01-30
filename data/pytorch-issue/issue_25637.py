import torch

@torch.jit.script
class Foo(object):
    def __init__(self, value: torch.Tensor):
        self.value = value

    def __getitem__(self, item: torch.Tensor):
        updated_value = self.value[item]
        return Foo(updated_value)

@torch.jit.script 
def bar(v: Foo, index: torch.Tensor):
    return v[index]

if __name__ == "__main__":
    a = torch.tensor([0,1,2,3,4,5])
    slice_index = torch.tensor([0,2,5])
    b = Foo(a)
    b_slice = b[slice_index]  # this one works
    print(b_slice.value)
    print(bar(b, slice_index).value) # this one does not work with @torch.jit.script