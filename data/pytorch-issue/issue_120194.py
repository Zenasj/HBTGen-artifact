import torch

exported = torch.export.export(model, example_inputs)
torch._inductor.aoti_compile_and_package(exported, package_path="model.pt2")

...

model = torch._inductor.aoti_load_package("model.pt2")
model(*example_inputs)