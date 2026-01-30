import torch

torch.onnx.export(conv, y, path, export_params=True, training=False)

# traning should take TrainingModel, not bool