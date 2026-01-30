import torch

model.model_body[0].auto_model = torch.compile(model.model_body[0].auto_model, backend="torch_tensorrt", dynamic=False,
                                options={"truncate_long_and_double": True,
                                         "precision": torch.half,
                                         "debug": True,
                                         "min_block_size": 1,
                                         "optimization_level": 4,
                                         "use_python_runtime": False})

model.model_body[0].auto_model = torch.compile(model.model_body[0].auto_model, mode="reduce-overhead")