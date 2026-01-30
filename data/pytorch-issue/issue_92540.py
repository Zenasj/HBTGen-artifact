import torch.nn as nn

import os
from einops import rearrange
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

def recursive_to(data, device):
    """Recursive move leaves to device for list/dict."""
    if isinstance(data, dict):
        return {key: recursive_to(val, device) for key, val in data.items()}
    if isinstance(data, (list, tuple)):
        return [recursive_to(elem, device) for elem in data]
    if isinstance(data, torch.Tensor):
        return data.to(device)
    return data

class QueryToImageSimpleONNXable(nn.Module):
    def __init__(self, dim, droprate):
        super().__init__()
        self.dropout = nn.Dropout(droprate)
        self.norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, dim, bias=True)

    def forward(
        self,
        query_content,
        query_position,
        key_content,
        key_position,
        key_size: Tuple[int],
    ):
        selected = query_position["mask"].any(dim=1)  # (M,)
        updated_query_content = query_content
        updated_query_content[selected] = torch.rand_like(query_content)[selected]
        return updated_query_content


def build_model(dim):
    model = QueryToImageSimpleONNXable(dim=dim, droprate=0.1)
    return model

def create_input(batch_size, dim, query_grid_size, image_size):
    M = torch.prod(torch.tensor(query_grid_size))
    N = torch.prod(torch.tensor(image_size))
    model_input = {
        "query_content": torch.rand(M, batch_size, dim),
        "query_position": {
            "raw": torch.rand(M, batch_size, 2),
            "mask": torch.rand(M, batch_size) > 0.3,
        },
        "key_content": torch.rand(N, batch_size, dim),
        "key_position": None,
        "key_size": image_size,
    }
    return model_input

def main():
    model = build_model(dim = 128)
    model_input = create_input(
        batch_size = 2,
        dim = 128,
        query_grid_size = (64, 64, 4),
        image_size = (15, 27),
    )
    model = model.cuda()
    model_input = recursive_to(model_input, "cuda")
    out = model(**model_input)
    input_names = ["actual_input_1"]
    output_names = ["output1"]
    torch.onnx.export(
        model,
        model_input,
        os.path.join("experiments", "onnx_test", "onnx_test_00.onnx"),
        verbose=True,
        input_names=input_names,
        output_names=output_names,
        opset_version=13,
    )


if __name__ == "__main__":
    main()