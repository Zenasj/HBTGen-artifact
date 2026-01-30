import torch

lightSpechModel_dynamo = torch.compile(task.model,backend="onnxrt_cpu",fullgraph=True)
model_exp = dynamo.export(lightSpechModel_dynamo,phone )