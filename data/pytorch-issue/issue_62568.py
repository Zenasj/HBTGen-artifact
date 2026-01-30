import torch

# ...
# do something
# ...

torch.onnx.export(model, inputs, path,
                        opset_version=9,
                        example_outputs=None,
                        input_names=["X"], output_names=["Y"],
                        custom_opsets={"my_domain": 1},
                        verbose=False)