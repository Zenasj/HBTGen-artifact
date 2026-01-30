import torch

filename_model = '/path/to/serialized/model'
model = torch.load(filename_model, map_location='cpu').eval()
traced_script_module = torch.jit.script(model)
traced_script_module.save("./exported_model.th")

forward()

filename_model = '/path/to/serialized/model'
model = torch.load(filename_model, map_location='cpu').eval()
example = some_example_input
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("./exported_model.th")

model.my_for_loop_module = torhc.jit.script(model.my_for_loop_module)
scripted_module = torch.jit.trace(model, example)

ans

onset

frame