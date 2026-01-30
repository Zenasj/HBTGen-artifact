import torch

def fn():
    x = [1,2,3]
    sliced_x = x[::-1]
    return sliced_x

scripted_fn = torch.jit.script(fn)
# Show the IR (intermediate representation) of the TorchScript code
print(scripted_fn.graph)

# Run the function in Python
print(fn())

# Run the function using the TorchScript interpreter (this fails)
print(scripted_fn())