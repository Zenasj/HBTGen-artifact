import torch.nn as nn

import torch
from typing import List

@torch.jit.script
def spatial_shapes_helper(embeds: List[torch.FloatTensor]) -> torch.LongTensor:
    return torch.tensor([
        [embeds[0].shape[2], embeds[0].shape[3]],
        [embeds[1].shape[2], embeds[1].shape[3]],
        [embeds[2].shape[2], embeds[2].shape[3]],
    ], device=embeds[0].device)

class MyShaphesModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        input_embeds0,
        input_embeds1,
        input_embeds2,
    ):
        input_embeds = [input_embeds0, input_embeds1, input_embeds2]
        spatial_shapes: torch.LongTensor = spatial_shapes_helper(input_embeds)
        offset_normalizer: torch.LongTensor = torch.flip(spatial_shapes, [1])
        
        return spatial_shapes, offset_normalizer

def test_myshapes():
    model = MyShaphesModule()

    embeds = tuple([
        torch.rand([1, 256, 37, 58]),
        torch.rand([1, 256, 74, 116]),
        torch.rand([1, 256, 147, 232])
    ])

    output = model(*embeds)

    input_names = ["embeds0", "embeds1", "embeds2"]
    output_names = [
        "spatial_shapes",
        "offset_normalizer"
    ]
    torch.onnx.export(
        model,
        embeds,
        "myshapes.onnx",
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        opset_version=17,
        dynamic_axes={
            "embeds0": [2,3],
            "embeds1": [2,3],
            "embeds2": [2,3],
        },
    )

if __name__ == "__main__":
    test_myshapes()
    print("done")