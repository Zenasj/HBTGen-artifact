import torch.nn.functional as F

import torch
import torchvision.transforms.functional as F

# 0. load model
model = torch.hub.load("pytorch/vision:v0.9.0", "mobilenet_v2", pretrained=True)
model = model.eval()

# 1. trace
model1 = torch.jit.trace(model, example_inputs=[
    F.normalize(
        torch.rand(1, 3, 224, 224),
        mean=torch.tensor([0.485, 0.456, 0.406]),
        std=torch.tensor([0.229, 0.224, 0.225])
    )
])
print(model1.graph)

# 2. freeze
model1 = model1.eval()  # otherwise, model1.training is undefined and torch.jit.freeze fails
model2 = torch.jit.freeze(model1)
print(model2.graph)  # inlined constants produce longer flattened graph, as expected

# 3. save both models
model1.save("model1.pt")
model2.save("model2.pt")

# 4. load the model that was not frozen - works as expected
model1_loaded = torch.jit.load("model1.pt")
print(model1_loaded.graph)  # equal to model1.graph, as expected

# 5. load the frozen model
model2_loaded = torch.jit.load("model2.pt")
print(model2_loaded.graph)  # NOT equal to model2.graph, why?
# note that this graph is roughly 2x longer than model2.graph due to many repeated patterns

# 6. attempt to fix the loaded frozen model:
model2_loaded = model2_loaded.eval()
print(model2_loaded.graph)  # still NOT equal to model2.graph, so eval is not the cause

model2_loaded_refrozen = torch.jit.freeze(model2_loaded)
print(model2_loaded_refrozen.graph)  # only now the graph is equal to model2.graph, why?