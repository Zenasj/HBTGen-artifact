import torch

_, _, tags = torch._C._get_operation_overload(schema.name, schema.overload_name)