import torch.nn as nn

# Opt 1
with element_wise_grad(model):
    model(inp).sum().backward()

    for p in model.parameter():
        print(p.grad_sample) # RuntimeError
        print(p.grad is not None) # has per-sample gradient

for p in model.parameter():
    print(p.grad_sample is not None)
    print(p.grad is None)



# Opt 2
out = nn.util.per_sample_call(model, inp)
out.sum().backward()

for p in model.parameter():
    print(p.grad_sample is not None)
    print(p.grad is None)


# Opt 3
some_dict = model.state_dict()
for k in some_dict:
    some_dict[k] = PerSampleObj(some_dict[k])

out = nn.util.stateless.functional_call(model, some_dict, inp)
out.sum().backward()

for k in some_dict:
    print(some_dict[k].grad)
for p in model.parameter():
    print(p.grad is None)
    
# Opt 4
model.per_sample_model_()
out = model(input)
out.sum().backward()

for p in model.parameter():
    print(p.grad_sample is not None)
    print(p.grad is None)