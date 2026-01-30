import torch

def custom_searchsorted(self,sorted_sequence, values, right=False, side=None, sorter=None):
    if side is not None:
        if side == 'left':
            right = False
        elif side == 'right':
            right = True
        else:
            raise ValueError("side must be 'left' or 'right'")
    
    if sorter is not None:
        sorted_sequence = torch.gather(sorted_sequence, -1, sorter)
    
    values_expanded = values.unsqueeze(-1)
    
    if right:
        mask = sorted_sequence > values_expanded
    else:
        mask = sorted_sequence >= values_expanded

    indices = torch.argmax(mask.int(), dim=-1)
    
    any_mask = torch.any(mask, dim=-1)
    last_dim = sorted_sequence.size(-1)
    indices = torch.where(any_mask, indices, last_dim)
    
    return indices

with torch.no_grad():
    torch.onnx.export(
        model,
        torch.from_numpy(audio).cuda(),
        "./AllInOneYoudao.onnx",
        input_names=["audio",],
        output_names=["output",],
        dynamic_axes={
            "audio": {0: "seq"},
            # "padding_mask": {0: "seq"},
            "output": {0:  "seq"},
        },
        opset_version=17,
        do_constant_folding=True,
    )