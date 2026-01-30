import torch
import torchvision

# An instance of your model.
model = torchvision.models.resnet18()

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 3, 224, 224).cuda()

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
model.cuda()

before = model(example)
print(before.shape)

traced_script_module = torch.jit.trace(model, example)
traced_script_module.save('hard.pt')

load_module = torch.jit.load("hard.pt")
after = load_module(example)

print(torch.sum(before - after))